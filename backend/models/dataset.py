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
from typing import Dict, Iterator, List, Optional, Tuple

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


_WIN_EXPR = (
    "CASE WHEN (pg.side = 'Blue' AND m.team1_side = 'Blue' AND m.winner = 1) "
    "       OR (pg.side = 'Blue' AND m.team2_side = 'Blue' AND m.winner = 2) "
    "       OR (pg.side = 'Red'  AND m.team1_side = 'Red'  AND m.winner = 1) "
    "       OR (pg.side = 'Red'  AND m.team2_side = 'Red'  AND m.winner = 2) "
    "  THEN 1 ELSE 0 END"
)


def _team_player_form(conn: sqlite3.Connection, team: str, before: str) -> Dict[str, float]:
    """Roster-aggregate form for the team's players, prior to `before`.

    `pg.win` is unreliable in our ingestion (often NULL), so derive wins via
    the matches join on side.
    """
    row = conn.execute(
        f"""
        SELECT
            COUNT(*) AS games,
            SUM({_WIN_EXPR}) AS wins,
            AVG(pg.kills) AS k,
            AVG(pg.deaths) AS d,
            AVG(pg.assists) AS a
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.team = :t
          AND m.datetime_utc < :before
        """,
        {"t": team, "before": before},
    ).fetchone()
    g = row["games"] or 0
    w = row["wins"] or 0
    deaths = row["d"] or 0
    kda = ((row["k"] or 0) + (row["a"] or 0)) / max(deaths, 1.0)
    return {
        "p_games":   float(g),
        "p_winrate": (w / g) if g else 0.0,
        "p_kda":     float(kda),
    }


def _team_champion_form(
    conn: sqlite3.Connection,
    game_id: str,
    before: str,
    league: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """For each side of `game_id`, compute aggregate champion-pool stats prior to
    `before`:
      - global_wr:           average global WR of each champion picked
      - league_wr:           average WR of each champion picked, filtered to `league`
                             when given (with a global fallback when in-league sample
                             is too thin)
      - player_on_champ_wr:  average of (this player's WR on this champion)
      - matchup_wr:          average WR of (champion1 vs same-role champion2),
                             team1 perspective
    """
    picks = conn.execute(
        """
        SELECT pg.player, pg.team, pg.side, pg.role, pg.champion
        FROM player_games pg
        WHERE pg.game_id = :g AND pg.role IS NOT NULL AND pg.champion IS NOT NULL
        """,
        {"g": game_id},
    ).fetchall()
    if not picks:
        empty = {"global_wr": 0.5, "league_wr": 0.5, "player_on_champ_wr": 0.5, "matchup_wr": 0.5, "n": 0.0}
        return {"team1": dict(empty), "team2": dict(empty)}

    # Identify which side is team1 vs team2 for this match.
    m_row = conn.execute(
        "SELECT team1_side FROM matches WHERE game_id = :g", {"g": game_id},
    ).fetchone()
    t1_side = m_row["team1_side"] if m_row else "Blue"

    by_team: Dict[str, list] = {"team1": [], "team2": []}
    by_role: Dict[str, dict] = {}
    for p in picks:
        bucket = "team1" if p["side"] == t1_side else "team2"
        by_team[bucket].append({"player": p["player"], "champion": p["champion"], "role": p["role"]})
        by_role.setdefault(p["role"], {})[bucket] = p["champion"]

    out: Dict[str, Dict[str, float]] = {}
    for bucket, picks_list in by_team.items():
        gw_sum = 0.0; gw_n = 0
        lw_sum = 0.0; lw_n = 0
        pw_sum = 0.0; pw_n = 0
        for sel in picks_list:
            r = conn.execute(
                f"""
                SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.champion = :c AND m.datetime_utc < :before
                """,
                {"c": sel["champion"], "before": before},
            ).fetchone()
            g, w = r["g"] or 0, r["w"] or 0
            if g >= 3:
                gw_sum += w / g; gw_n += 1
            # League-filtered WR with cross-league fallback when in-league is sparse.
            if league:
                rl = conn.execute(
                    f"""
                    SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                    FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                    WHERE pg.champion = :c
                      AND m.datetime_utc < :before
                      AND m.league_code = :lg
                    """,
                    {"c": sel["champion"], "before": before, "lg": league},
                ).fetchone()
                gl, wl = rl["g"] or 0, rl["w"] or 0
                if gl >= 5:
                    lw_sum += wl / gl; lw_n += 1
                elif g >= 3:
                    lw_sum += w / g; lw_n += 1
            elif g >= 3:
                lw_sum += w / g; lw_n += 1
            r2 = conn.execute(
                f"""
                SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = :p AND pg.champion = :c AND m.datetime_utc < :before
                """,
                {"p": sel["player"], "c": sel["champion"], "before": before},
            ).fetchone()
            g2, w2 = r2["g"] or 0, r2["w"] or 0
            if g2 >= 1:
                pw_sum += w2 / g2; pw_n += 1
        out[bucket] = {
            "global_wr": (gw_sum / gw_n) if gw_n else 0.5,
            "league_wr": (lw_sum / lw_n) if lw_n else 0.5,
            "player_on_champ_wr": (pw_sum / pw_n) if pw_n else 0.5,
            "n": float(len(picks_list)),
        }

    # Per-role champion matchup WR (team1 perspective).
    mu_sum = 0.0; mu_n = 0
    for role, sides in by_role.items():
        c1 = sides.get("team1"); c2 = sides.get("team2")
        if not (c1 and c2):
            continue
        # WR of c1 when facing c2 in same game (any role on either side, not just same role,
        # would conflate signals — but we only have lineup not pick-side mapping by role for
        # historical games, so use "c1's WR in games where c2 was on the opposing team").
        r = conn.execute(
            f"""
            SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
            FROM player_games pg
            JOIN matches m ON m.game_id = pg.game_id
            JOIN player_games opp ON opp.game_id = pg.game_id
                                 AND opp.side != pg.side
            WHERE pg.champion = :c1
              AND opp.champion = :c2
              AND m.datetime_utc < :before
            """,
            {"c1": c1, "c2": c2, "before": before},
        ).fetchone()
        g, w = r["g"] or 0, r["w"] or 0
        if g >= 3:
            mu_sum += w / g; mu_n += 1
    matchup_wr = (mu_sum / mu_n) if mu_n else 0.5
    out["team1"]["matchup_wr"] = matchup_wr
    out["team2"]["matchup_wr"] = 1.0 - matchup_wr if mu_n else 0.5
    return out


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
            cf  = _team_champion_form(conn, m["game_id"], before, league=m["league_code"])

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
                "team1_champ_global_wr": cf["team1"]["global_wr"],
                "team2_champ_global_wr": cf["team2"]["global_wr"],
                "team1_champ_league_wr": cf["team1"]["league_wr"],
                "team2_champ_league_wr": cf["team2"]["league_wr"],
                "team1_pchamp_wr":       cf["team1"]["player_on_champ_wr"],
                "team2_pchamp_wr":       cf["team2"]["player_on_champ_wr"],
                "champ_matchup_wr":      cf["team1"]["matchup_wr"],
                "wr_diff":           tf1["winrate"] - tf2["winrate"],
                "p_wr_diff":         pf1["p_winrate"] - pf2["p_winrate"],
                "side_wr_diff":      t1_side_wr - t2_side_wr,
                "champ_global_diff": cf["team1"]["global_wr"] - cf["team2"]["global_wr"],
                "champ_league_diff": cf["team1"]["league_wr"] - cf["team2"]["league_wr"],
                "pchamp_diff":       cf["team1"]["player_on_champ_wr"] - cf["team2"]["player_on_champ_wr"],
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
    "team1_champ_global_wr", "team2_champ_global_wr",
    "team1_champ_league_wr", "team2_champ_league_wr",
    "team1_pchamp_wr", "team2_pchamp_wr", "champ_matchup_wr",
    "wr_diff", "p_wr_diff", "side_wr_diff",
    "champ_global_diff", "champ_league_diff", "pchamp_diff",
]


if __name__ == "__main__":
    import json
    rows = build_training_rows()
    print(f"Built {len(rows)} training rows")
    if rows:
        print("First row:", json.dumps(rows[0], indent=2))
        print("Last row:",  json.dumps(rows[-1], indent=2))
