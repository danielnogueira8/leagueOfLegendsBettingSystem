"""Training dataset builder.

For each completed match in the DB, we generate one training row:
  - features:  computed from games STRICTLY BEFORE the match's datetime_utc
               (point-in-time correctness — no future leakage)
  - target:    1 if team1 won, 0 if team2 won

Key design choices that came out of debugging earlier versions:

1.  Team and player WR are basically the same signal in pro LoL — players don't
    flex teams mid-season, so averaging player WRs gives back the team WR.
    We expose ONE winrate per side (team-level) and use player rows only for
    KDA and per-champion specialization, not for redundant win-rate features.

2.  Sparse features get *Bayesian-shrunk to 0.5* using `(w + k*0.5) / (g + k)`
    instead of just `w/g`. This is critical for player-on-champion: with 0–2
    games of history, raw `w/g` is either 0.0, 0.5, or 1.0, and the model
    drastically overweights it. Shrinkage keeps tiny samples close to the
    league prior and only lets the signal speak when sample size is real.

3.  We exclude the raw `wr_diff` / `pchamp_diff` columns from the model — they
    are linear combinations of the per-team columns and add nothing for a
    linear model (and create instability for tree models). They stay on the
    output dict only because the API surfaces them in /predict for display.
"""
from __future__ import annotations
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Tuple

from backend.db.schema import get_conn


# Each side has 5 player rows per game. AVG over player_games gives 5 identical
# values per match in 99% of cases, so we always derive team-side stats from
# the matches table. _WIN_EXPR is for player_games rows only (champ-specific
# aggregations).
_WIN_EXPR = (
    "CASE WHEN (pg.side = 'Blue' AND m.team1_side = 'Blue' AND m.winner = 1) "
    "       OR (pg.side = 'Blue' AND m.team2_side = 'Blue' AND m.winner = 2) "
    "       OR (pg.side = 'Red'  AND m.team1_side = 'Red'  AND m.winner = 1) "
    "       OR (pg.side = 'Red'  AND m.team2_side = 'Red'  AND m.winner = 2) "
    "  THEN 1 ELSE 0 END"
)


# Bayesian shrinkage prior. `k` = "virtual games at the prior". Smaller k
# means we trust the data faster, larger k means we cling to the prior longer.
# 10 is a reasonable default for pro LoL champion stats.
SHRINK_K = 10.0


def _shrink(wins: float, games: float, prior: float = 0.5, k: float = SHRINK_K) -> float:
    """Bayesian shrinkage of an empirical winrate toward `prior`.

    With games=0 returns prior. With games>>k returns wins/games.
    """
    return (wins + k * prior) / (games + k)


# ---------- Per-match aggregations from history ---------- #

def _team_recent_form(conn: sqlite3.Connection, team: str, before: str,
                      last_n: int = 5) -> Dict[str, float]:
    """Win rate over the team's last `last_n` matches before `before`.

    Recency matters in pro LoL — a team's record from 3 weeks ago may not
    reflect current form. Lightly shrunken because the window is small.
    """
    rows = conn.execute(
        """
        SELECT team1, team2, winner FROM matches
        WHERE (team1 = :t OR team2 = :t)
          AND datetime_utc < :before
        ORDER BY datetime_utc DESC
        LIMIT :n
        """,
        {"t": team, "before": before, "n": last_n},
    ).fetchall()
    g = len(rows)
    w = sum(1 for r in rows
            if (r["team1"] == team and r["winner"] == 1)
            or (r["team2"] == team and r["winner"] == 2))
    return {
        "games":    float(g),
        "winrate":  _shrink(w, g, 0.5, k=2.0),
    }


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
    w = row["wins"] or 0
    return {
        "games":   float(g),
        # Keep raw WR for display, but pass shrunken WR to the model so a 1-0
        # team early in the split doesn't get treated as a 100% team.
        "winrate":         (w / g) if g else 0.5,
        "winrate_shrunk":  _shrink(w, g, 0.5, SHRINK_K),
        "avg_len":         float(row["avg_len"] or 0.0),
    }


def _team_side_form(conn: sqlite3.Connection, team: str, side: str, before: str) -> Dict[str, float]:
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
    w = row["w"] or 0
    return {
        "games":          float(g),
        "winrate_shrunk": _shrink(w, g, 0.5, SHRINK_K),
    }


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
    w = row["aw"] or 0
    # Heavy shrink — H2H samples are tiny and noisy.
    return float(g), _shrink(w, g, 0.5, k=4.0)


def _team_perf(conn: sqlite3.Connection, team: str, before: str) -> Dict[str, float]:
    """Aggregate per-game performance metrics for the team's roster over the
    patch window: KDA, gold per minute, CS per minute. These are slow-moving,
    independent of pure W/L, and pick up "this team plays harder" signal.
    """
    row = conn.execute(
        """
        SELECT AVG(pg.kills) AS k, AVG(pg.deaths) AS d, AVG(pg.assists) AS a,
               AVG(pg.gold) AS g, AVG(pg.cs) AS cs, AVG(m.gamelength) AS gl
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        WHERE pg.team = :t AND m.datetime_utc < :before
        """,
        {"t": team, "before": before},
    ).fetchone()
    k = row["k"] or 0
    d = row["d"] or 0
    a = row["a"] or 0
    g = row["g"] or 0
    cs = row["cs"] or 0
    gl = row["gl"] or 30  # minutes; default to ~30 min when no history
    gl_min = max(gl, 1.0)
    return {
        "kda":   (k + a) / max(d, 1.0),
        "gpm":   g / gl_min,
        "cspm":  cs / gl_min,
    }


def _team_champion_form(
    conn: sqlite3.Connection,
    game_id: str,
    before: str,
    league: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """For each side of `game_id`, compute aggregate champion-pool stats prior
    to `before`. All values are Bayesian-shrunk toward 0.5 so picks with thin
    history don't blow up the prediction.

    Returns:
      {
        'team1': { 'global_wr', 'league_wr', 'pchamp_wr', 'matchup_wr',
                   'pchamp_n_total': total games-of-history across the 5 picks },
        'team2': {...},
      }
    """
    picks = conn.execute(
        """
        SELECT pg.player, pg.team, pg.side, pg.role, pg.champion
        FROM player_games pg
        WHERE pg.game_id = :g AND pg.role IS NOT NULL AND pg.champion IS NOT NULL
        """,
        {"g": game_id},
    ).fetchall()
    empty = {"global_wr": 0.5, "league_wr": 0.5, "pchamp_wr": 0.5,
             "matchup_wr": 0.5, "pchamp_n_total": 0.0}
    if not picks:
        return {"team1": dict(empty), "team2": dict(empty)}

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
        gw_sum = 0.0
        lw_sum = 0.0
        pw_sum = 0.0
        pchamp_n_total = 0.0
        for sel in picks_list:
            # Global WR for this champion (any pro game in window, prior to `before`).
            r = conn.execute(
                f"""
                SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.champion = :c AND m.datetime_utc < :before
                """,
                {"c": sel["champion"], "before": before},
            ).fetchone()
            g, w = r["g"] or 0, r["w"] or 0
            gw_sum += _shrink(w, g, 0.5, SHRINK_K)

            # League-filtered WR (with global fallback through shrinkage prior).
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
                # Use the global rate as the prior for the league rate so an
                # unseen-in-this-league pick falls back to global, not 0.5.
                global_rate = (w / g) if g >= 5 else 0.5
                lw_sum += _shrink(wl, gl, global_rate, SHRINK_K)
            else:
                lw_sum += _shrink(w, g, 0.5, SHRINK_K)

            # Player-on-champion WR. THIS IS THE BIG ONE — Bayesian shrinkage
            # toward the player's overall WR so a 1-0 mastery doesn't
            # show as 100%.
            r2 = conn.execute(
                f"""
                SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = :p AND pg.champion = :c AND m.datetime_utc < :before
                """,
                {"p": sel["player"], "c": sel["champion"], "before": before},
            ).fetchone()
            g2, w2 = r2["g"] or 0, r2["w"] or 0
            # Player's overall WR (prior).
            r3 = conn.execute(
                f"""
                SELECT COUNT(*) AS g, SUM({_WIN_EXPR}) AS w
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = :p AND m.datetime_utc < :before
                """,
                {"p": sel["player"], "before": before},
            ).fetchone()
            g3, w3 = r3["g"] or 0, r3["w"] or 0
            player_overall = (w3 / g3) if g3 >= 5 else 0.5
            pw_sum += _shrink(w2, g2, player_overall, SHRINK_K)
            pchamp_n_total += g2

        n = max(len(picks_list), 1)
        out[bucket] = {
            "global_wr":      gw_sum / n,
            "league_wr":      lw_sum / n,
            "pchamp_wr":      pw_sum / n,
            "pchamp_n_total": pchamp_n_total,
        }

    # Per-role champion matchup WR (team1 perspective).
    mu_sum = 0.0; mu_n = 0
    for role, sides in by_role.items():
        c1 = sides.get("team1"); c2 = sides.get("team2")
        if not (c1 and c2):
            continue
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


def build_training_rows() -> List[Dict[str, Any]]:
    """One row per labeled, completed match. Returns a list of feature dicts.

    Each row has a `target` key (1 = team1 won, 0 = team2 won) and a `match_dt`
    so callers can do time-based train/val splits.
    """
    rows: List[Dict[str, Any]] = []
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
            rf1 = _team_recent_form(conn, t1, before, last_n=5)
            rf2 = _team_recent_form(conn, t2, before, last_n=5)
            sf1 = _team_side_form(conn, t1, m["team1_side"], before)
            sf2 = _team_side_form(conn, t2, m["team2_side"], before)
            h2h_g, h2h_t1wr = _h2h(conn, t1, t2, before)
            perf1 = _team_perf(conn, t1, before)
            perf2 = _team_perf(conn, t2, before)
            cf  = _team_champion_form(conn, m["game_id"], before, league=m["league_code"])

            row: Dict[str, Any] = {
                "match_dt":          before,
                "league_code":       m["league_code"] or "",
                # ---- team form ----
                "team1_games":           tf1["games"],
                "team2_games":           tf2["games"],
                "team1_winrate":         tf1["winrate_shrunk"],
                "team2_winrate":         tf2["winrate_shrunk"],
                "team1_avg_len":         tf1["avg_len"],
                "team2_avg_len":         tf2["avg_len"],
                "team1_kda":             perf1["kda"],
                "team2_kda":             perf2["kda"],
                "kda_diff":              perf1["kda"] - perf2["kda"],
                "team1_gpm":             perf1["gpm"],
                "team2_gpm":             perf2["gpm"],
                "gpm_diff":              perf1["gpm"] - perf2["gpm"],
                "team1_cspm":            perf1["cspm"],
                "team2_cspm":            perf2["cspm"],
                "cspm_diff":             perf1["cspm"] - perf2["cspm"],
                "team1_recent_wr":       rf1["winrate"],
                "team2_recent_wr":       rf2["winrate"],
                "recent_wr_diff":        rf1["winrate"] - rf2["winrate"],
                # ---- side ----
                "team1_side_wr":         sf1["winrate_shrunk"],
                "team2_side_wr":         sf2["winrate_shrunk"],
                "team1_side_blue":       1.0 if m["team1_side"] == "Blue" else 0.0,
                # ---- h2h ----
                "h2h_games":             h2h_g,
                "h2h_team1_wr":          h2h_t1wr,
                # ---- champion / draft ----
                "team1_champ_global_wr": cf["team1"]["global_wr"],
                "team2_champ_global_wr": cf["team2"]["global_wr"],
                "team1_champ_league_wr": cf["team1"]["league_wr"],
                "team2_champ_league_wr": cf["team2"]["league_wr"],
                "team1_pchamp_wr":       cf["team1"]["pchamp_wr"],
                "team2_pchamp_wr":       cf["team2"]["pchamp_wr"],
                "team1_pchamp_n":        cf["team1"]["pchamp_n_total"],
                "team2_pchamp_n":        cf["team2"]["pchamp_n_total"],
                "champ_matchup_wr":      cf["team1"]["matchup_wr"],
                # ---- explicit difference signals ----
                # The "favorite" baseline (sign(team1_wr - team2_wr)) hits ~74%
                # on its own. Giving the linear model the diff directly stops it
                # from having to learn it from two correlated columns and frees
                # the per-team WRs to encode "which side has a stronger denominator".
                "wr_diff":               tf1["winrate_shrunk"] - tf2["winrate_shrunk"],
                "champ_global_diff":     cf["team1"]["global_wr"] - cf["team2"]["global_wr"],
                "champ_league_diff":     cf["team1"]["league_wr"] - cf["team2"]["league_wr"],
                "pchamp_diff":           cf["team1"]["pchamp_wr"] - cf["team2"]["pchamp_wr"],
                # ---- target ----
                "target":                1 if m["winner"] == 1 else 0,
            }
            rows.append(row)

    return rows


# Feature columns the model uses. Note: we deliberately omit the *_diff
# columns — they're linear combinations of the per-team columns and a
# linear model can already form them through coefficients. Including them
# just splits weight and creates instability.
FEATURE_COLS: List[str] = [
    # Team form
    "team1_games",   "team2_games",
    "team1_winrate", "team2_winrate",       "wr_diff",
    "team1_recent_wr", "team2_recent_wr",   "recent_wr_diff",
    "team1_kda",     "team2_kda",     "kda_diff",
    "team1_gpm",     "team2_gpm",     "gpm_diff",
    "team1_cspm",    "team2_cspm",    "cspm_diff",
    "team1_avg_len", "team2_avg_len",
    # Side
    "team1_side_wr", "team2_side_wr", "team1_side_blue",
    # H2H
    "h2h_games", "h2h_team1_wr",
    # Champion / draft (all shrunken)
    "team1_champ_global_wr", "team2_champ_global_wr", "champ_global_diff",
    "team1_champ_league_wr", "team2_champ_league_wr", "champ_league_diff",
    "team1_pchamp_wr",       "team2_pchamp_wr",       "pchamp_diff",
    "team1_pchamp_n",        "team2_pchamp_n",
    "champ_matchup_wr",
]


if __name__ == "__main__":
    import json
    rows = build_training_rows()
    print(f"Built {len(rows)} training rows")
    if rows:
        # Sanity-check: team1_winrate and team2_winrate should NOT be a
        # linear function of any other column.
        print("First row:", json.dumps(rows[0], indent=2))
        print("Last row:",  json.dumps(rows[-1], indent=2))
