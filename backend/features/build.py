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
        "winrate":    wr,
        "blue_wr":    (side_blue["w"] or 0) / (side_blue["g"] or 1) if side_blue["g"] else 0.0,
        "red_wr":     (side_red["w"]  or 0) / (side_red["g"]  or 1) if side_red["g"] else 0.0,
        "avg_len":    float(row["avg_len"] or 0.0),
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
    return {
        "h2h_games":   float(g),
        "h2h_team1_wr": (row["a_wins"] or 0) / g if g else 0.5,
    }


def _player_recent(conn: sqlite3.Connection, player: str) -> Dict[str, float]:
    row = conn.execute(
        """
        SELECT
            COUNT(*)        AS games,
            AVG(win)        AS wr,
            AVG(kills)      AS k,
            AVG(deaths)     AS d,
            AVG(assists)    AS a
        FROM player_games
        WHERE player = :p
        """,
        {"p": player},
    ).fetchone()
    g = row["games"] or 0
    deaths = row["d"] or 0
    kda = ((row["k"] or 0) + (row["a"] or 0)) / max(deaths, 1.0)
    return {
        "games":   float(g),
        "winrate": float(row["wr"] or 0.0),
        "kda":     float(kda),
    }


def _player_champ(conn: sqlite3.Connection, player: str, champion: str) -> Dict[str, float]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS games, AVG(win) AS wr
        FROM player_games
        WHERE player = :p AND champion = :c
        """,
        {"p": player, "c": champion},
    ).fetchone()
    g = row["games"] or 0
    return {
        "champ_games":   float(g),
        "champ_winrate": float(row["wr"] or 0.0),
    }


def build_features(inp: MatchInput) -> Dict[str, float]:
    """Build a flat feature dict for the given matchup."""
    feats: Dict[str, float] = {}
    with get_conn() as conn:
        for prefix, team in (("team1", inp.team1), ("team2", inp.team2)):
            tr = _team_recent(conn, team)
            for k, v in tr.items():
                feats[f"{prefix}_{k}"] = v

        h = _h2h(conn, inp.team1, inp.team2)
        feats.update(h)

        feats["team1_side_blue"] = 1.0 if inp.team1_side == "Blue" else 0.0

        for prefix, players in (("team1", inp.team1_players or []), ("team2", inp.team2_players or [])):
            wr_sum = 0.0
            kda_sum = 0.0
            cwr_sum = 0.0
            cwr_n = 0
            for sel in players:
                pr = _player_recent(conn, sel.player)
                pc = _player_champ(conn, sel.player, sel.champion)
                wr_sum  += pr["winrate"]
                kda_sum += pr["kda"]
                if pc["champ_games"] > 0:
                    cwr_sum += pc["champ_winrate"]
                    cwr_n   += 1
            n = max(len(players), 1)
            feats[f"{prefix}_player_wr_avg"]   = wr_sum / n
            feats[f"{prefix}_player_kda_avg"]  = kda_sum / n
            feats[f"{prefix}_player_champ_wr"] = (cwr_sum / cwr_n) if cwr_n else 0.5
            feats[f"{prefix}_player_champ_n"]  = float(cwr_n)

    feats["wr_diff"]     = feats["team1_winrate"] - feats["team2_winrate"]
    feats["pwr_diff"]    = feats["team1_player_wr_avg"] - feats["team2_player_wr_avg"]
    feats["champ_diff"]  = feats["team1_player_champ_wr"] - feats["team2_player_champ_wr"]
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
