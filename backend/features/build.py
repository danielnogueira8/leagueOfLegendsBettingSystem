"""Feature engineering for match win-probability prediction.

Given a (team1, team2, patch, optional player→champion mapping) input, build
a feature vector from competitive-only history in SQLite.

Design notes:
- Only use rows in player_games / matches that exist in the DB. Since ingestion
  only retains the last N patches, all features are implicitly "recent meta".
- Be robust to sparse data: when a player has no games on a champion in window,
  fall back gracefully (overall form replaces champ-specific signal).

Feature groups:
  TEAM:   recent winrate, side winrate, avg game length, head-to-head record
  PLAYER: recent winrate, KDA average, champion-specific winrate (if any),
          games played in window
  META:   side (blue/red) global winrate on each patch
"""
from __future__ import annotations
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from backend.db.schema import get_conn


@dataclass
class PlayerSelection:
    player: str
    champion: str
    role: Optional[str] = None  # Top/Jungle/Mid/Bot/Support — optional, looked up if missing


@dataclass
class MatchInput:
    team1: str
    team2: str
    patch: Optional[str] = None             # None => use most recent patch in window
    team1_side: str = "Blue"                # 'Blue' or 'Red'
    team1_players: Optional[List[PlayerSelection]] = None
    team2_players: Optional[List[PlayerSelection]] = None
    league: Optional[str] = None            # None => auto-detect from team1's most-played league


# Bayesian shrinkage prior used at inference time. Must match the value in
# backend.models.dataset (SHRINK_K = 10) for live features to fall in the same
# distribution as the training features.
SHRINK_K = 10.0


def _shrink(wins: float, games: float, prior: float = 0.5, k: float = SHRINK_K) -> float:
    return (wins + k * prior) / (games + k)


def _team_recent_form(conn: sqlite3.Connection, team: str, last_n: int = 5) -> float:
    """Shrunken WR over the team's last N matches in the patch window.
    Mirrors the training-time signal so live and training distributions match.
    """
    rows = conn.execute(
        """
        SELECT team1, team2, winner FROM matches
        WHERE (team1 = :t OR team2 = :t)
        ORDER BY datetime_utc DESC
        LIMIT :n
        """,
        {"t": team, "n": last_n},
    ).fetchall()
    g = len(rows)
    w = sum(1 for r in rows
            if (r["team1"] == team and r["winner"] == 1)
            or (r["team2"] == team and r["winner"] == 2))
    return _shrink(w, g, 0.5, k=2.0)


def _team_perf(conn: sqlite3.Connection, team: str) -> Dict[str, float]:
    """Per-minute KPIs (kda/gpm/cspm) for a team across the window. Mirrors
    backend.models.dataset._team_perf for live consistency."""
    row = conn.execute(
        """
        SELECT AVG(pg.kills) AS k, AVG(pg.deaths) AS d, AVG(pg.assists) AS a,
               AVG(pg.gold) AS g, AVG(pg.cs) AS cs, AVG(m.gamelength) AS gl
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.team = :t
        """,
        {"t": team},
    ).fetchone()
    k = row["k"] or 0; d = row["d"] or 0; a = row["a"] or 0
    g = row["g"] or 0; cs = row["cs"] or 0; gl = row["gl"] or 30
    gl_min = max(gl, 1.0)
    return {"kda": (k + a) / max(d, 1.0), "gpm": g / gl_min, "cspm": cs / gl_min}


def _team_recent(conn: sqlite3.Connection, team: str) -> Dict[str, float]:
    """Win rate, side win rates, avg gamelength for a team across the window."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS games,
            SUM(CASE WHEN (team1 = :t AND winner = 1) OR (team2 = :t AND winner = 2) THEN 1 ELSE 0 END) AS wins,
            AVG(gamelength) AS avg_len
        FROM matches
        WHERE team1 = :t OR team2 = :t
        """,
        {"t": team},
    ).fetchone()
    games = row["games"] or 0
    wins = row["wins"] or 0
    wr = wins / games if games else 0.0

    side_blue = conn.execute(
        """
        SELECT
            COUNT(*) AS g,
            SUM(CASE WHEN winner = 1 THEN 1 ELSE 0 END) AS w
        FROM matches
        WHERE (team1 = :t AND team1_side = 'Blue') OR (team2 = :t AND team2_side = 'Blue')
        """,
        {"t": team},
    ).fetchone()
    side_red = conn.execute(
        """
        SELECT
            COUNT(*) AS g,
            SUM(CASE WHEN
                (team1 = :t AND winner = 1 AND team1_side = 'Red') OR
                (team2 = :t AND winner = 2 AND team2_side = 'Red')
            THEN 1 ELSE 0 END) AS w
        FROM matches
        WHERE (team1 = :t AND team1_side = 'Red') OR (team2 = :t AND team2_side = 'Red')
        """,
        {"t": team},
    ).fetchone()

    return {
        "games":      float(games),
        # `winrate_raw` is the "what fans would say" stat shown in the UI;
        # `winrate` is shrunken and is what the model consumes.
        "winrate_raw":  wr,
        "winrate":      _shrink(wins, games),
        "blue_wr":      _shrink(side_blue["w"] or 0, side_blue["g"] or 0),
        "red_wr":       _shrink(side_red["w"]  or 0, side_red["g"]  or 0),
        "blue_wr_raw":  (side_blue["w"] or 0) / (side_blue["g"] or 1) if side_blue["g"] else 0.0,
        "red_wr_raw":   (side_red["w"]  or 0) / (side_red["g"]  or 1) if side_red["g"] else 0.0,
        "avg_len":      float(row["avg_len"] or 0.0),
    }


def _h2h(conn: sqlite3.Connection, team1: str, team2: str) -> Dict[str, float]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS g,
            SUM(CASE WHEN
                (team1 = :a AND team2 = :b AND winner = 1) OR
                (team1 = :b AND team2 = :a AND winner = 2)
            THEN 1 ELSE 0 END) AS a_wins
        FROM matches
        WHERE (team1 = :a AND team2 = :b) OR (team1 = :b AND team2 = :a)
        """,
        {"a": team1, "b": team2},
    ).fetchone()
    g = row["g"] or 0
    w = row["a_wins"] or 0
    return {
        "h2h_games":     float(g),
        "h2h_team1_wr":  _shrink(w, g, 0.5, k=4.0),  # match dataset.py
        "h2h_team1_raw": (w / g) if g else 0.5,
    }


_WIN_EXPR = (
    "CASE WHEN (pg.side = 'Blue' AND m.team1_side = 'Blue' AND m.winner = 1) "
    "       OR (pg.side = 'Blue' AND m.team2_side = 'Blue' AND m.winner = 2) "
    "       OR (pg.side = 'Red'  AND m.team1_side = 'Red'  AND m.winner = 1) "
    "       OR (pg.side = 'Red'  AND m.team2_side = 'Red'  AND m.winner = 2) "
    "  THEN 1 ELSE 0 END"
)


def _player_recent(conn: sqlite3.Connection, player: str) -> Dict[str, float]:
    """Player's WR and KDA across the patch window. We derive wins via the
    matches join because pg.win is unreliable in our ingestion."""
    row = conn.execute(
        f"""
        SELECT
            COUNT(*)        AS games,
            SUM({_WIN_EXPR}) AS wins,
            AVG(pg.kills)   AS k,
            AVG(pg.deaths)  AS d,
            AVG(pg.assists) AS a
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.player = :p
        """,
        {"p": player},
    ).fetchone()
    g = row["games"] or 0
    w = row["wins"] or 0
    deaths = row["d"] or 0
    kda = ((row["k"] or 0) + (row["a"] or 0)) / max(deaths, 1.0)
    return {
        "games":   float(g),
        "winrate": (w / g) if g else 0.0,
        "kda":     float(kda),
    }


def _player_champ(conn: sqlite3.Connection, player: str, champion: str) -> Dict[str, float]:
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS games, SUM({_WIN_EXPR}) AS wins
        FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.player = :p AND pg.champion = :c
        """,
        {"p": player, "c": champion},
    ).fetchone()
    g = row["games"] or 0
    w = row["wins"] or 0
    return {
        "champ_games":   float(g),
        "champ_winrate": (w / g) if g else 0.0,
    }


def _champion_global(
    conn: sqlite3.Connection,
    champion: str,
    league: Optional[str] = None,
) -> Dict[str, float]:
    """Global WR for a champion. When `league` is given, filters to that league.

    Returns games + winrate. Caller decides whether the sample is large enough
    to trust (e.g. 5+ games) and falls back to the cross-league number if not.
    """
    sql = f"""
        SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
        FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.champion = :c
    """
    params: Dict[str, object] = {"c": champion}
    if league:
        sql += " AND m.league_code = :lg"
        params["lg"] = league
    row = conn.execute(sql, params).fetchone()
    g = row["g"] or 0
    w = row["w"] or 0
    return {"games": float(g), "winrate": (w / g) if g else 0.5}


def _team_primary_league(conn: sqlite3.Connection, team: str) -> Optional[str]:
    """League where the team plays most often within the patch window."""
    row = conn.execute(
        """
        SELECT league_code, COUNT(*) AS g
        FROM matches
        WHERE team1 = :t OR team2 = :t
        GROUP BY league_code
        ORDER BY g DESC
        LIMIT 1
        """,
        {"t": team},
    ).fetchone()
    return row["league_code"] if row and row["league_code"] else None


def _champion_matchup(conn: sqlite3.Connection, c1: str, c2: str) -> Dict[str, float]:
    """WR of c1 in games where c2 was on the opposing team. Symmetric: 1 - this
    is c2's WR vs c1."""
    row = conn.execute(
        f"""
        SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        JOIN player_games opp ON opp.game_id = pg.game_id
                             AND opp.side != pg.side
        WHERE pg.champion = :c1 AND opp.champion = :c2
        """,
        {"c1": c1, "c2": c2},
    ).fetchone()
    g = row["g"] or 0
    w = row["w"] or 0
    return {"games": float(g), "winrate": (w / g) if g else 0.5}


def build_features(inp: MatchInput) -> Dict[str, float]:
    """Build a flat feature dict for the given matchup."""
    feats: Dict[str, float] = {}
    with get_conn() as conn:
        for prefix, team in (("team1", inp.team1), ("team2", inp.team2)):
            tr = _team_recent(conn, team)
            for k, v in tr.items():
                feats[f"{prefix}_{k}"] = v
            feats[f"{prefix}_recent_wr"] = _team_recent_form(conn, team, last_n=5)
            perf = _team_perf(conn, team)
            feats[f"{prefix}_kda"]  = perf["kda"]
            feats[f"{prefix}_gpm"]  = perf["gpm"]
            feats[f"{prefix}_cspm"] = perf["cspm"]

        h = _h2h(conn, inp.team1, inp.team2)
        feats.update(h)

        feats["team1_side_blue"] = 1.0 if inp.team1_side == "Blue" else 0.0

        # Resolve the league context for champion-pool WR. Caller can override;
        # otherwise we infer from team1 (teams almost always share their league).
        league = inp.league or _team_primary_league(conn, inp.team1)
        feats["league_context"] = league or ""

        team1_picks = inp.team1_players or []
        team2_picks = inp.team2_players or []
        for prefix, players in (("team1", team1_picks), ("team2", team2_picks)):
            wr_sum = 0.0
            kda_sum = 0.0
            cwr_sum = 0.0
            gwr_sum = 0.0
            lwr_sum = 0.0
            cwr_n_raw = 0          # picks where the player has any history on this champ
            pchamp_n_total = 0.0   # total games of player-on-champ history across the 5 picks
            for sel in players:
                pr = _player_recent(conn, sel.player)
                pc = _player_champ(conn, sel.player, sel.champion)
                cg = _champion_global(conn, sel.champion)
                wr_sum  += pr["winrate"]
                kda_sum += pr["kda"]

                # Player-on-champ: shrink toward the player's overall WR (or 0.5
                # if the player has <5 games of any kind). With shrinkage, a 1-0
                # pick on a new champ shows ~player_overall, not 100%.
                player_prior = pr["winrate"] if pr["games"] >= 5 else 0.5
                cwr_sum += _shrink(
                    pc["champ_games"] * pc["champ_winrate"],  # implicit "wins"
                    pc["champ_games"], player_prior,
                )
                cwr_n_raw += 1 if pc["champ_games"] > 0 else 0
                pchamp_n_total += pc["champ_games"]

                # Champion global WR — shrunken toward 0.5.
                gwr_sum += _shrink(
                    cg["games"] * cg["winrate"], cg["games"], 0.5,
                )

                # League-filtered WR. Use global rate as prior when in-league
                # sample is thin, so a champ with no league history falls back to
                # global rather than to 0.5.
                global_rate = cg["winrate"] if cg["games"] >= 5 else 0.5
                if league:
                    in_league = _champion_global(conn, sel.champion, league=league)
                    lwr_sum += _shrink(
                        in_league["games"] * in_league["winrate"],
                        in_league["games"], global_rate,
                    )
                else:
                    lwr_sum += _shrink(
                        cg["games"] * cg["winrate"], cg["games"], 0.5,
                    )

            # When no players were provided, fall back to the training-mean
            # neutral value (0.5 for shrunken winrates) instead of dividing 0/1
            # and producing 0.0 — that would look like a wild outlier to the
            # standardized model and produce spurious extreme predictions.
            if players:
                n = len(players)
                feats[f"{prefix}_player_wr_avg"]    = wr_sum / n
                feats[f"{prefix}_player_kda_avg"]   = kda_sum / n
                feats[f"{prefix}_player_champ_wr"]  = cwr_sum / n
                feats[f"{prefix}_player_champ_n"]   = float(cwr_n_raw)
                feats[f"{prefix}_pchamp_n_total"]   = float(pchamp_n_total)
                feats[f"{prefix}_champ_global_wr"]  = gwr_sum / n
                feats[f"{prefix}_champ_league_wr"]  = lwr_sum / n
            else:
                feats[f"{prefix}_player_wr_avg"]    = 0.5
                feats[f"{prefix}_player_kda_avg"]   = 3.0   # rough roster-average KDA
                feats[f"{prefix}_player_champ_wr"]  = 0.5
                feats[f"{prefix}_player_champ_n"]   = 0.0
                feats[f"{prefix}_pchamp_n_total"]   = 0.0
                feats[f"{prefix}_champ_global_wr"]  = 0.5
                feats[f"{prefix}_champ_league_wr"]  = 0.5

        # Per-role champion-vs-champion matchup WR (team1 perspective).
        by_role_t1 = {p.role: p.champion for p in team1_picks if p.role and p.champion}
        by_role_t2 = {p.role: p.champion for p in team2_picks if p.role and p.champion}
        mu_sum = 0.0; mu_n = 0
        for role, c1 in by_role_t1.items():
            c2 = by_role_t2.get(role)
            if not c2:
                continue
            mu = _champion_matchup(conn, c1, c2)
            if mu["games"] >= 3:
                mu_sum += mu["winrate"]; mu_n += 1
        feats["champ_matchup_wr"] = (mu_sum / mu_n) if mu_n else 0.5

    feats["wr_diff"]           = feats["team1_winrate"] - feats["team2_winrate"]
    feats["pwr_diff"]          = feats["team1_player_wr_avg"] - feats["team2_player_wr_avg"]
    feats["champ_diff"]        = feats["team1_player_champ_wr"] - feats["team2_player_champ_wr"]
    feats["champ_global_diff"] = feats["team1_champ_global_wr"] - feats["team2_champ_global_wr"]
    feats["champ_league_diff"] = feats["team1_champ_league_wr"] - feats["team2_champ_league_wr"]
    return feats


def baseline_probability(feats: Dict[str, float]) -> float:
    """Heuristic baseline P(team1 wins) for sanity-checking before we have a model.

    Linear blend of team form, head-to-head, side, and per-player champ winrate.
    Coefficients are hand-tuned, not learned. This is replaced by a real model later.
    """
    z = 0.0
    z += 2.5 * feats.get("wr_diff", 0.0)
    z += 1.0 * (feats.get("h2h_team1_wr", 0.5) - 0.5)
    z += 0.6 * (feats.get("team1_side_blue", 0.0) - 0.5) * (feats.get("team1_blue_wr", 0.5) - feats.get("team2_red_wr", 0.5))
    z += 1.5 * feats.get("pwr_diff", 0.0)
    z += 1.0 * feats.get("champ_diff", 0.0)
    # Logistic.
    import math
    return 1.0 / (1.0 + math.exp(-z))


if __name__ == "__main__":
    # Quick demo: lookup a recent matchup once data exists.
    inp = MatchInput(team1="T1", team2="Gen.G", patch=None, team1_side="Blue")
    f = build_features(inp)
    print({k: round(v, 4) for k, v in f.items()})
    print("baseline P(T1 wins):", round(baseline_probability(f), 4))
