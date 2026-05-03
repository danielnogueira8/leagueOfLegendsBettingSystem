"""Polymarket Gamma API client.

We use the public read-only Gamma API (no auth needed) to list LoL events
and pull current outcome prices. Each "event" usually has many sub-markets
(series winner, individual game winners, kill totals, baron/dragon props).
For our value-bet flow we only care about the **match winner** markets:
    * Series winner (Bo3 / Bo5)
    * Per-game winner (Game 1 / Game 2 / Game 3)

Polymarket charges ~2% on resolution, so we filter edges below 5pp by default
to keep a margin of safety.
"""
from __future__ import annotations
import json
import re
import requests
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from backend.config import USER_AGENT

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 20

# Markets whose `question` field matches one of these regexes are the
# "head-to-head winner" markets we can model. Other markets (kill totals,
# baron props) require different models.
SERIES_RE = re.compile(r"^LoL: .+ vs .+ \(BO\d\)", re.IGNORECASE)
GAME_RE   = re.compile(r"- Game (\d) Winner$", re.IGNORECASE)


@dataclass
class MarketTeam:
    """Team-side line on a Polymarket market."""
    name: str          # Polymarket's spelling, e.g. "Gen G"
    price: float       # Implied prob, 0..1


@dataclass
class HeadToHeadMarket:
    event_title:    str
    market_question: str
    market_id:      str
    kind:           str            # "series" | "game"
    game_number:    Optional[int]  # 1/2/3 for kind=game, None for series
    bo:             int            # 3 or 5
    end_iso:        str
    league_hint:    Optional[str]  # "LCS Regular Season" etc, parsed from event title
    team1:          MarketTeam
    team2:          MarketTeam
    volume:         float
    liquidity:      float


def _get_json(path: str, params: Dict[str, Any]) -> Any:
    r = requests.get(
        GAMMA_BASE + path,
        params=params,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_open_lol_events() -> List[Dict[str, Any]]:
    """Currently-open LoL events on Polymarket, freshest endDate first."""
    raw = _get_json("/events", {
        "tag_slug": "league-of-legends",
        "limit": 200,
        "closed": "false",
    })
    now = datetime.now(timezone.utc)
    upcoming: List[Tuple[float, Dict[str, Any]]] = []
    for ev in raw:
        end_iso = ev.get("endDate")
        if not end_iso:
            continue
        try:
            end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
        except ValueError:
            continue
        if end_dt > now:
            upcoming.append((end_dt.timestamp(), ev))
    upcoming.sort(key=lambda x: x[0])
    return [e for _, e in upcoming]


def _parse_outcome_array(value: Any) -> List[str]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except (ValueError, TypeError):
            return []
    return []


def _league_hint_from_title(title: str) -> Optional[str]:
    """Extract the trailing league context from a Polymarket event title.
    Example: 'LoL: T1 vs HLE (BO3) - LCK Regular Season' -> 'LCK Regular Season'."""
    if " - " in title:
        return title.rsplit(" - ", 1)[1].strip()
    return None


def extract_h2h_markets(event: Dict[str, Any]) -> List[HeadToHeadMarket]:
    """Pull out the head-to-head winner markets from an event."""
    title = event.get("title", "")
    end_iso = event.get("endDate", "")
    league_hint = _league_hint_from_title(title)

    bo = 3
    m = re.search(r"\(BO(\d)\)", title)
    if m:
        bo = int(m.group(1))

    out: List[HeadToHeadMarket] = []
    for mk in event.get("markets", []):
        question = mk.get("question") or ""
        outcomes = _parse_outcome_array(mk.get("outcomes"))
        prices   = _parse_outcome_array(mk.get("outcomePrices"))
        if len(outcomes) != 2 or len(prices) != 2:
            continue
        try:
            p1, p2 = float(prices[0]), float(prices[1])
        except (TypeError, ValueError):
            continue
        # Filter to markets whose two outcomes look like team names (i.e. neither
        # is "Yes/No/Over/Under/Odd/Even/etc." — these are head-to-head markets).
        non_team_outcomes = {"yes", "no", "over", "under", "odd", "even"}
        if any(o.lower() in non_team_outcomes for o in outcomes):
            continue

        kind = "series"
        game_n: Optional[int] = None
        gm = GAME_RE.search(question)
        if gm:
            kind = "game"
            game_n = int(gm.group(1))
        elif SERIES_RE.search(question):
            kind = "series"
        else:
            # Skip handicap and other markets — those don't map cleanly to our
            # per-game probability.
            continue

        out.append(HeadToHeadMarket(
            event_title    = title,
            market_question= question,
            market_id      = str(mk.get("id") or mk.get("conditionId") or ""),
            kind           = kind,
            game_number    = game_n,
            bo             = bo,
            end_iso        = end_iso,
            league_hint    = league_hint,
            team1          = MarketTeam(outcomes[0], p1),
            team2          = MarketTeam(outcomes[1], p2),
            volume         = float(mk.get("volume") or 0.0),
            liquidity      = float(mk.get("liquidity") or 0.0),
        ))

    return out


def list_h2h_markets() -> List[HeadToHeadMarket]:
    """Convenience: every actionable head-to-head LoL market on Polymarket."""
    out: List[HeadToHeadMarket] = []
    for ev in fetch_open_lol_events():
        out.extend(extract_h2h_markets(ev))
    return out


# ---------- Series math ---------- #

def per_game_prob_to_series(p: float, bo: int) -> float:
    """Convert a per-game win probability into a Bo3/Bo5 series win probability.

    Assumes games are independent (a strong but standard assumption — pro teams
    rarely "tilt" badly between games of a series in our patch window).

    Bo3: win 2 before opponent wins 2.
    Bo5: win 3 before opponent wins 3.
    """
    if not 0.0 < p < 1.0:
        return float(p)
    q = 1.0 - p
    if bo == 3:
        # Win in 2-0 + win in 2-1
        return p*p + 2*p*p*q
    if bo == 5:
        # Win in 3-0 + 3-1 + 3-2
        return p**3 + 3*(p**3)*q + 6*(p**3)*(q**2)
    return p


if __name__ == "__main__":
    markets = list_h2h_markets()
    print(f"Found {len(markets)} actionable head-to-head LoL markets")
    for m in markets[:15]:
        kind_str = f"game{m.game_number}" if m.kind == "game" else f"series-Bo{m.bo}"
        print(f"  [{m.end_iso[:16]}] {kind_str:>10}  "
              f"{m.team1.name:>22}({m.team1.price:.2f}) vs "
              f"{m.team2.name:<22}({m.team2.price:.2f})  "
              f"vol=${m.volume:>6.0f}")
