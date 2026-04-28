"""Training dataset builder.

For each completed match in the DB, we generate one training row:
  - features:  computed from games STRICTLY BEFORE the match's datetime_utc
               (point-in-time correctness — no future leakage)
  - target:    1 if team1 won, 0 if team2 won

The features mirror what backend.features.build does for live prediction, but
restricted to history that pre-dates the match we're labeling. We avoid relying
on the player→champion mapping for now (predictable at inference time only when
the lineup is known).
"""
from __future__ import annotations
import sqlite3
from typing import Dict, Iterator, List, Tuple

from backend.db.schema import get_conn


# ---------- Per-match aggregations from history ---------- #

def _team_form(conn: sqlite3.Connection, team: str, before: str) -> Dict[str, float]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS games,
            SUM(CASE WHEN (team1 = :t AND winner = 1) OR (team2 = :t AND winner = 2) THEN 1 ELSE 0 END) AS wins,
            AVG(gamelength) AS avg_len
        FROM matches
        WHERE (team1 = :t OR team2 = :t)
          AND datetime_utc < :before
        """,
        {"t": team, "before": before},
    ).fetchone()
    g = row["games"] or 0
    return {
        "games":   float(g),
        "winrate": (row["wins"] or 0) / g if g else 0.0,
        "avg_len": float(row["avg_len"] or 0.0),
    }


def _team_side_form(conn: sqlite3.Connection, team: str, side: str, before: str) -> float:
    """Side-specific historical winrate for `team` on `side` ('Blue'/'Red')."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS g,
            SUM(CASE WHEN
                (team1 = :t AND winner = 1) OR
                (team2 = :t AND winner = 2)
            THEN 1 ELSE 0 END) AS w
        FROM matches
        WHERE datetime_utc < :before
          AND (
            (team1 = :t AND team1_side = :s) OR
            (team2 = :t AND team2_side = :s)
          )
        """,
        {"t": team, "s": side, "before": before},
    ).fetchone()
    g = row["g"] or 0
    return (row["w"] or 0) / g if g else 0.0


def _h2h(conn: sqlite3.Connection, a: str, b: str, before: str) -> Tuple[float, float]:
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS g,
            SUM(CASE WHEN
                (team1 = :a AND team2 = :b AND winner = 1) OR
                (team1 = :b AND team2 = :a AND winner = 2)
            THEN 1 ELSE 0 END) AS aw
        FROM matches
        WHERE datetime_utc < :before
          AND ((team1 = :a AND team2 = :b) OR (team1 = :b AND team2 = :a))
        """,
        {"a": a, "b": b, "before": before},
    ).fetchone()
    g = row["g"] or 0
    return float(g), ((row["aw"] or 0) / g if g else 0.5)


def _team_player_form(conn: sqlite3.Connection, team: str, before: str) -> Dict[str, float]:
    """Roster-aggregate form for the team's players, prior to `before`."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) AS games,
            AVG(win) AS wr,
            AVG(kills) AS k,
            AVG(deaths) AS d,
            AVG(assists) AS a
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.team = :t
          AND m.datetime_utc < :before
        """,
        {"t": team, "before": before},
    ).fetchone()
    g = row["games"] or 0
    deaths = row["d"] or 0
    kda = ((row["k"] or 0) + (row["a"] or 0)) / max(deaths, 1.0)
    return {
        "p_games":   float(g),
        "p_winrate": float(row["wr"] or 0.0),
        "p_kda":     float(kda),
    }


def build_training_rows() -> List[Dict[str, float]]:
    """One row per labeled, completed match. Returns a list of feature dicts.

    Each row has a `target` key (1 = team1 won, 0 = team2 won) and a `match_dt`
    so callers can do time-based train/val splits.
    """
    rows: List[Dict[str, float]] = []
    with get_conn() as conn:
        matches = conn.execute(
            """
            SELECT game_id, datetime_utc, team1, team2, winner,
                   team1_side, team2_side, league_code, patch
            FROM matches
            WHERE winner IN (1, 2)
              AND team1_side IS NOT NULL
              AND team2_side IS NOT NULL
            ORDER BY datetime_utc ASC
            """
        ).fetchall()

        for m in matches:
            t1, t2 = m["team1"], m["team2"]
            before = m["datetime_utc"]
            tf1 = _team_form(conn, t1, before)
            tf2 = _team_form(conn, t2, before)
            t1_side_wr = _team_side_form(conn, t1, m["team1_side"], before)
            t2_side_wr = _team_side_form(conn, t2, m["team2_side"], before)
            h2h_g, h2h_t1wr = _h2h(conn, t1, t2, before)
            pf1 = _team_player_form(conn, t1, before)
            pf2 = _team_player_form(conn, t2, before)

            row = {
                "match_dt":          before,
                "league_code":       m["league_code"] or "",
                "team1_games":       tf1["games"],
                "team1_winrate":     tf1["winrate"],
                "team1_avg_len":     tf1["avg_len"],
                "team2_games":       tf2["games"],
                "team2_winrate":     tf2["winrate"],
                "team2_avg_len":     tf2["avg_len"],
                "team1_side_wr":     t1_side_wr,
                "team2_side_wr":     t2_side_wr,
                "team1_side_blue":   1.0 if m["team1_side"] == "Blue" else 0.0,
                "h2h_games":         h2h_g,
                "h2h_team1_wr":      h2h_t1wr,
                "team1_p_games":     pf1["p_games"],
                "team1_p_winrate":   pf1["p_winrate"],
                "team1_p_kda":       pf1["p_kda"],
                "team2_p_games":     pf2["p_games"],
                "team2_p_winrate":   pf2["p_winrate"],
                "team2_p_kda":       pf2["p_kda"],
                "wr_diff":           tf1["winrate"] - tf2["winrate"],
                "p_wr_diff":         pf1["p_winrate"] - pf2["p_winrate"],
                "side_wr_diff":      t1_side_wr - t2_side_wr,
                "target":            1 if m["winner"] == 1 else 0,
            }
            rows.append(row)

    return rows


# Feature columns the model uses (excludes meta cols match_dt / league_code / target).
FEATURE_COLS: List[str] = [
    "team1_games", "team1_winrate", "team1_avg_len",
    "team2_games", "team2_winrate", "team2_avg_len",
    "team1_side_wr", "team2_side_wr", "team1_side_blue",
    "h2h_games", "h2h_team1_wr",
    "team1_p_games", "team1_p_winrate", "team1_p_kda",
    "team2_p_games", "team2_p_winrate", "team2_p_kda",
    "wr_diff", "p_wr_diff", "side_wr_diff",
]


if __name__ == "__main__":
    import json
    rows = build_training_rows()
    print(f"Built {len(rows)} training rows")
    if rows:
        print("First row:", json.dumps(rows[0], indent=2))
        print("Last row:",  json.dumps(rows[-1], indent=2))
