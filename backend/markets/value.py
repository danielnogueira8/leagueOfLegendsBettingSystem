"""Value-bet engine.

For each open Polymarket head-to-head market we:
  1. Resolve the team names to teams in our DB (fuzzy match).
  2. Run our model to get a per-game probability.
  3. Convert it to series prob if the market is a series winner.
  4. Compute edge (model − market), EV, and Kelly stake.

Most of the complexity is **honest team-name matching** — Polymarket spells
teams like "Gen G", we store "Gen.G"; "FlyQuest" vs "Flyquest"; etc.
"""
from __future__ import annotations
import re
import sqlite3
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from backend.db.schema import get_conn
from backend.features.build import MatchInput, build_features
from backend.markets.polymarket import (
    HeadToHeadMarket, list_h2h_markets, per_game_prob_to_series,
)
from backend.models.train import load_model


def _strip_team(s: str) -> str:
    """Aggressively normalize a team name for fuzzy matching."""
    s = s.lower()
    # Common suffixes that vary between sources.
    for suf in [" esports", " gaming", " e-sports", " esport", " club", " team"]:
        s = s.replace(suf, "")
    # Drop dots, dashes, spaces, common punctuation.
    s = re.sub(r"[\s\.\-_'\"\(\)/&]", "", s)
    return s


@lru_cache(maxsize=1)
def _all_teams() -> List[str]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT DISTINCT team FROM ("
            "  SELECT team1 AS team FROM matches UNION SELECT team2 FROM matches"
            ") WHERE team IS NOT NULL"
        ).fetchall()
    return [r["team"] for r in rows]


def resolve_team(market_name: str) -> Optional[str]:
    """Best-match an in-DB team name for the given Polymarket team string.

    Returns None if no match crosses the similarity threshold — caller should
    skip the market in that case (we can't model a team we have no data on).
    """
    target = _strip_team(market_name)
    best, best_score = None, 0.0
    for t in _all_teams():
        score = SequenceMatcher(None, target, _strip_team(t)).ratio()
        if score > best_score:
            best, best_score = t, score
    return best if best_score >= 0.85 else None


@dataclass
class ValueBet:
    event_title:     str
    market_question: str
    kind:            str          # "series" | "game"
    game_number:     Optional[int]
    bo:              int
    end_iso:         str
    league_hint:     Optional[str]

    favored_team:        str       # the side we want to bet on
    favored_market_name: str       # how Polymarket spells it (for the link)
    opponent_team:       str

    model_prob:    float          # P(favored_team wins this market), per our model
    market_prob:   float
    edge_pp:       float          # model_prob - market_prob, in [−1, 1]
    ev_per_dollar: float          # Expected return per $1 stake (incl. losing stake)
    kelly_pct:     float          # Optimal bankroll fraction per Kelly
    quarter_kelly_pct: float      # Conservative fractional Kelly

    market_volume:    float
    market_liquidity: float


def _kelly(p: float, market_prob: float) -> float:
    """Kelly stake fraction. p = your prob, market_prob = implied prob.

    Decimal odds = 1 / market_prob, so net winnings per $1 stake = (1/market_prob) − 1
    Kelly = (p × b − q) / b, where b = (1/market_prob − 1) and q = 1 − p.
    Returns 0 if Kelly is negative (don't bet against yourself).
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    b = (1.0 / market_prob) - 1.0
    if b <= 0:
        return 0.0
    k = (p * b - (1 - p)) / b
    return max(k, 0.0)


def evaluate_market(
    mkt: HeadToHeadMarket,
    min_edge: float = 0.05,
    min_volume: float = 500.0,
) -> Optional[ValueBet]:
    t1 = resolve_team(mkt.team1.name)
    t2 = resolve_team(mkt.team2.name)
    if not t1 or not t2 or t1 == t2:
        return None
    if mkt.volume < min_volume:
        return None

    bundle = load_model()
    if bundle is None:
        return None
    model = bundle["model"]
    cols  = bundle["feature_cols"]

    # Side is unknown pre-game on Polymarket. Use Blue for team1 — minor effect
    # on the model and consistent across both perspectives.
    inp = MatchInput(team1=t1, team2=t2, team1_side="Blue")
    feats = build_features(inp)

    # Map live features to model features. Inlined here to avoid a circular
    # import of server._model_features_from_live, which lives in the API layer.
    side_wr_t1 = feats.get("team1_blue_wr", 0.5)
    side_wr_t2 = feats.get("team2_red_wr",  0.5)
    t1_wr = feats.get("team1_winrate", 0.5)
    t2_wr = feats.get("team2_winrate", 0.5)
    t1_cg = feats.get("team1_champ_global_wr", 0.5)
    t2_cg = feats.get("team2_champ_global_wr", 0.5)
    t1_cl = feats.get("team1_champ_league_wr", 0.5)
    t2_cl = feats.get("team2_champ_league_wr", 0.5)
    t1_pc = feats.get("team1_player_champ_wr", 0.5)
    t2_pc = feats.get("team2_player_champ_wr", 0.5)
    mfeats: Dict[str, float] = {
        "team1_games":           feats.get("team1_games", 0.0),
        "team2_games":           feats.get("team2_games", 0.0),
        "team1_winrate":         t1_wr,
        "team2_winrate":         t2_wr,
        "wr_diff":               t1_wr - t2_wr,
        "team1_recent_wr":       feats.get("team1_recent_wr", 0.5),
        "team2_recent_wr":       feats.get("team2_recent_wr", 0.5),
        "recent_wr_diff":        feats.get("team1_recent_wr", 0.5) - feats.get("team2_recent_wr", 0.5),
        "team1_kda":             feats.get("team1_kda", 0.0),
        "team2_kda":             feats.get("team2_kda", 0.0),
        "kda_diff":              feats.get("team1_kda", 0.0) - feats.get("team2_kda", 0.0),
        "team1_gpm":             feats.get("team1_gpm", 0.0),
        "team2_gpm":             feats.get("team2_gpm", 0.0),
        "gpm_diff":              feats.get("team1_gpm", 0.0) - feats.get("team2_gpm", 0.0),
        "team1_cspm":            feats.get("team1_cspm", 0.0),
        "team2_cspm":            feats.get("team2_cspm", 0.0),
        "cspm_diff":             feats.get("team1_cspm", 0.0) - feats.get("team2_cspm", 0.0),
        "team1_avg_len":         feats.get("team1_avg_len", 0.0),
        "team2_avg_len":         feats.get("team2_avg_len", 0.0),
        "team1_side_wr":         side_wr_t1,
        "team2_side_wr":         side_wr_t2,
        "team1_side_blue":       1.0,
        "h2h_games":             feats.get("h2h_games", 0.0),
        "h2h_team1_wr":          feats.get("h2h_team1_wr", 0.5),
        "team1_champ_global_wr": t1_cg,
        "team2_champ_global_wr": t2_cg,
        "champ_global_diff":     t1_cg - t2_cg,
        "team1_champ_league_wr": t1_cl,
        "team2_champ_league_wr": t2_cl,
        "champ_league_diff":     t1_cl - t2_cl,
        "team1_pchamp_wr":       t1_pc,
        "team2_pchamp_wr":       t2_pc,
        "pchamp_diff":           t1_pc - t2_pc,
        "team1_pchamp_n":        feats.get("team1_pchamp_n_total", 0.0),
        "team2_pchamp_n":        feats.get("team2_pchamp_n_total", 0.0),
        "champ_matchup_wr":      feats.get("champ_matchup_wr", 0.5),
    }
    # In case some columns aren't in mfeats (e.g. older models), fill 0.
    x = [[mfeats.get(c, 0.0) for c in cols]]
    p_t1_game = float(model.predict_proba(x)[0][1])

    # Convert per-game prob to per-market prob.
    if mkt.kind == "series":
        p_t1_market = per_game_prob_to_series(p_t1_game, mkt.bo)
    else:
        p_t1_market = p_t1_game  # per-game market

    # Decide which side has positive edge.
    edge_t1 = p_t1_market - mkt.team1.price
    edge_t2 = (1 - p_t1_market) - mkt.team2.price

    if edge_t1 >= edge_t2 and edge_t1 >= min_edge:
        favored = (mkt.team1.name, t1, mkt.team2.name, t2 if t2 else mkt.team2.name)
        model_prob, market_prob = p_t1_market, mkt.team1.price
    elif edge_t2 > edge_t1 and edge_t2 >= min_edge:
        favored = (mkt.team2.name, t2, mkt.team1.name, t1 if t1 else mkt.team1.name)
        model_prob, market_prob = 1 - p_t1_market, mkt.team2.price
    else:
        return None

    edge_pp = model_prob - market_prob
    ev_per_dollar = (model_prob / market_prob) - 1.0  # net return per $1 stake
    k = _kelly(model_prob, market_prob)

    return ValueBet(
        event_title     = mkt.event_title,
        market_question = mkt.market_question,
        kind            = mkt.kind,
        game_number     = mkt.game_number,
        bo              = mkt.bo,
        end_iso         = mkt.end_iso,
        league_hint     = mkt.league_hint,
        favored_team    = favored[1],
        favored_market_name = favored[0],
        opponent_team   = favored[3],
        model_prob      = round(model_prob, 4),
        market_prob     = round(market_prob, 4),
        edge_pp         = round(edge_pp, 4),
        ev_per_dollar   = round(ev_per_dollar, 4),
        kelly_pct       = round(k * 100, 2),
        quarter_kelly_pct = round(k * 25, 2),
        market_volume    = mkt.volume,
        market_liquidity = mkt.liquidity,
    )


def find_value_bets(
    min_edge: float = 0.05,
    min_volume: float = 500.0,
    kinds: Optional[List[str]] = None,
) -> List[Dict]:
    """Run the full pipeline. Returns a list of dicts sorted by edge (best first)."""
    out: List[ValueBet] = []
    for mkt in list_h2h_markets():
        if kinds and mkt.kind not in kinds:
            continue
        vb = evaluate_market(mkt, min_edge=min_edge, min_volume=min_volume)
        if vb is not None:
            out.append(vb)
    out.sort(key=lambda v: -v.edge_pp)
    return [asdict(v) for v in out]


if __name__ == "__main__":
    import json
    bets = find_value_bets(min_edge=0.05, min_volume=500.0)
    print(f"Found {len(bets)} value bets (edge ≥ 5pp, volume ≥ $500)")
    for b in bets[:20]:
        kind_str = f"game{b['game_number']}" if b['kind'] == 'game' else f"Bo{b['bo']}"
        print(f"  [{b['end_iso'][:16]}] {kind_str:>5}  "
              f"{b['favored_team']:>22} "
              f"model={b['model_prob']*100:>5.1f}% market={b['market_prob']*100:>5.1f}% "
              f"edge={b['edge_pp']*100:+.1f}pp  "
              f"EV={b['ev_per_dollar']*100:+.0f}%  "
              f"¼Kelly={b['quarter_kelly_pct']:.1f}%")
