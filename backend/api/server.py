"""FastAPI server exposing prediction + lookup endpoints.

Run:
    uvicorn backend.api.server:app --reload --port 8000

Endpoints:
  GET  /health
  GET  /patches                 -> patches in window + which is current
  GET  /teams?league=LCK        -> teams seen in window (optionally filter)
  GET  /players?team=T1         -> players seen for a team (defaults: most recent role)
  GET  /matches/recent?team=T1  -> recent matches involving a team
  POST /predict                 -> heuristic probability for a matchup
"""
from __future__ import annotations
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import json
import time
from functools import lru_cache
from pathlib import Path

import requests as _requests

from backend.config import USER_AGENT, DATA_DIR
from backend.db.schema import get_conn, init_db
from backend.features.build import (
    MatchInput, PlayerSelection, build_features, baseline_probability,
)
from backend.models.train import load_model
from backend.models.dataset import FEATURE_COLS

# Ensure the SQLite schema exists at boot. On a fresh deploy with an empty
# mounted volume (e.g. Railway) the DB file doesn't exist yet — without this,
# every query 500s and the healthcheck never passes.
init_db()
print(f"[startup] DATA_DIR={DATA_DIR}", flush=True)

app = FastAPI(title="LoL Betting System", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerSelectionDTO(BaseModel):
    player: str
    champion: str
    role: Optional[str] = None


class PredictRequest(BaseModel):
    team1: str
    team2: str
    team1_side: str = "Blue"
    patch: Optional[str] = None
    league: Optional[str] = None  # If omitted, inferred from team1's primary league
    team1_players: Optional[List[PlayerSelectionDTO]] = None
    team2_players: Optional[List[PlayerSelectionDTO]] = None


# Tiny TTL cache for read-heavy endpoints. Cleared automatically after `_TTL`
# seconds. Reset by restarting the process or by running ingestion (which
# changes underlying data and would naturally serve a stale cache).
_CACHE: dict[str, tuple[float, object]] = {}
_TTL = 60  # seconds


def _cache_get(key: str):
    hit = _CACHE.get(key)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    return None


def _cache_put(key: str, value):
    _CACHE[key] = (time.monotonic() + _TTL, value)
    return value


def _model_metrics() -> dict | None:
    p = DATA_DIR / "model_metrics.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


@app.get("/health")
def health():
    with get_conn() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM matches").fetchone()["n"]
        n_pg = conn.execute("SELECT COUNT(*) AS n FROM player_games").fetchone()["n"]
        n_teams = conn.execute(
            "SELECT COUNT(DISTINCT team) AS n FROM ("
            "  SELECT team1 AS team FROM matches UNION SELECT team2 FROM matches"
            ") WHERE team IS NOT NULL"
        ).fetchone()["n"]
        last = conn.execute(
            "SELECT MAX(datetime_utc) AS dt FROM matches"
        ).fetchone()["dt"]
    metrics = _model_metrics()
    return {
        "ok": True,
        "matches_in_db": n,
        "player_rows":   n_pg,
        "teams":         n_teams,
        "latest_match":  last,
        "model": {
            "trained":     metrics is not None,
            "accuracy":    metrics["accuracy_val"] if metrics else None,
            "log_loss":    metrics["log_loss_val"] if metrics else None,
            "n_train":     metrics["n_train"] if metrics else None,
        } if metrics else {"trained": False},
    }


@app.get("/patches")
def patches():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT patch, is_current, in_window, fetched_at FROM patches ORDER BY patch DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/teams")
def teams(league: Optional[str] = None):
    sql = """
        SELECT DISTINCT team FROM (
            SELECT team1 AS team, league_code FROM matches
            UNION
            SELECT team2 AS team, league_code FROM matches
        ) WHERE team IS NOT NULL
    """
    params: list = []
    if league:
        sql += " AND league_code = ?"
        params.append(league)
    sql += " ORDER BY team"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [r["team"] for r in rows]


@app.get("/players")
def players(team: Optional[str] = Query(None)):
    sql = "SELECT name, team, role FROM players"
    params: list = []
    if team:
        sql += " WHERE team = ?"
        params.append(team)
    sql += " ORDER BY name"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/team/{team}/lineup")
def team_lineup(team: str):
    """Auto-detected starting lineup for a team.

    For each role, returns the player who started most often in that role for
    this team in the most recent match they played. If a role has multiple
    candidates we pick the one with the most recent start; ties go to highest
    games-played count in window.
    """
    with get_conn() as conn:
        # Confirm the team exists in our window.
        if not conn.execute(
            "SELECT 1 FROM matches WHERE team1 = ? OR team2 = ? LIMIT 1",
            (team, team),
        ).fetchone():
            raise HTTPException(404, f"No competitive history for team: {team!r}")

        rows = conn.execute(
            """
            SELECT pg.role, pg.player, COUNT(*) AS games, MAX(m.datetime_utc) AS last_played
            FROM player_games pg
            JOIN matches m ON m.game_id = pg.game_id
            WHERE pg.team = ? AND pg.role IS NOT NULL
            GROUP BY pg.role, pg.player
            ORDER BY pg.role, last_played DESC, games DESC
            """,
            (team,),
        ).fetchall()

    chosen: dict[str, dict] = {}
    for r in rows:
        role = r["role"]
        if role not in chosen:
            chosen[role] = {
                "role":  role,
                "player": r["player"],
                "games":  r["games"],
                "last_played": r["last_played"],
            }

    return {
        "team": team,
        "lineup": [chosen[r] for r in ("Top", "Jungle", "Mid", "Bot", "Support") if r in chosen],
    }


@app.get("/team/{team}/stats")
def team_stats(team: str):
    """Aggregate stats for a team across the patch window."""
    key = f"stats:{team}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    with get_conn() as conn:
        if not conn.execute(
            "SELECT 1 FROM matches WHERE team1 = ? OR team2 = ? LIMIT 1",
            (team, team),
        ).fetchone():
            raise HTTPException(404, f"No competitive history for team: {team!r}")
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS games,
                SUM(CASE WHEN (team1 = :t AND winner = 1) OR (team2 = :t AND winner = 2) THEN 1 ELSE 0 END) AS wins,
                AVG(gamelength) AS avg_len
            FROM matches
            WHERE (team1 = :t OR team2 = :t)
            """,
            {"t": team},
        ).fetchone()
        sides = conn.execute(
            """
            SELECT
                SUM(CASE WHEN (team1 = :t AND team1_side = 'Blue') OR (team2 = :t AND team2_side = 'Blue') THEN 1 ELSE 0 END) AS blue_g,
                SUM(CASE WHEN ((team1 = :t AND team1_side = 'Blue' AND winner = 1) OR (team2 = :t AND team2_side = 'Blue' AND winner = 2)) THEN 1 ELSE 0 END) AS blue_w,
                SUM(CASE WHEN (team1 = :t AND team1_side = 'Red') OR (team2 = :t AND team2_side = 'Red') THEN 1 ELSE 0 END) AS red_g,
                SUM(CASE WHEN ((team1 = :t AND team1_side = 'Red' AND winner = 1) OR (team2 = :t AND team2_side = 'Red' AND winner = 2)) THEN 1 ELSE 0 END) AS red_w
            FROM matches
            WHERE team1 = :t OR team2 = :t
            """,
            {"t": team},
        ).fetchone()
        recent = conn.execute(
            """
            SELECT game_id, datetime_utc, league_code, patch, team1, team2,
                   team1_side, team2_side, winner
            FROM matches
            WHERE team1 = ? OR team2 = ?
            ORDER BY datetime_utc DESC
            LIMIT 8
            """,
            (team, team),
        ).fetchall()

    g = row["games"] or 0
    w = row["wins"] or 0
    payload = {
        "team": team,
        "games": g,
        "wins": w,
        "winrate": w / g if g else 0.0,
        "avg_gamelength": float(row["avg_len"] or 0.0),
        "blue":  {"games": sides["blue_g"] or 0, "wins": sides["blue_w"] or 0,
                  "winrate": (sides["blue_w"] or 0) / (sides["blue_g"] or 1) if sides["blue_g"] else 0.0},
        "red":   {"games": sides["red_g"] or 0, "wins": sides["red_w"] or 0,
                  "winrate": (sides["red_w"] or 0) / (sides["red_g"] or 1) if sides["red_g"] else 0.0},
        "recent_matches": [dict(r) for r in recent],
    }
    return _cache_put(key, payload)


@app.get("/h2h")
def head_to_head(team1: str, team2: str):
    """Head-to-head record + recent meetings."""
    key = f"h2h:{team1}|{team2}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS g,
                SUM(CASE WHEN
                    (team1 = :a AND team2 = :b AND winner = 1) OR
                    (team1 = :b AND team2 = :a AND winner = 2)
                THEN 1 ELSE 0 END) AS team1_wins
            FROM matches
            WHERE (team1 = :a AND team2 = :b) OR (team1 = :b AND team2 = :a)
            """,
            {"a": team1, "b": team2},
        ).fetchone()
        recent = conn.execute(
            """
            SELECT game_id, datetime_utc, league_code, patch, team1, team2,
                   team1_side, team2_side, winner
            FROM matches
            WHERE (team1 = ? AND team2 = ?) OR (team1 = ? AND team2 = ?)
            ORDER BY datetime_utc DESC
            LIMIT 5
            """,
            (team1, team2, team2, team1),
        ).fetchall()

    g = row["g"] or 0
    payload = {
        "team1": team1,
        "team2": team2,
        "games": g,
        "team1_wins": row["team1_wins"] or 0,
        "team2_wins": (g - (row["team1_wins"] or 0)),
        "recent": [dict(r) for r in recent],
    }
    return _cache_put(key, payload)


@lru_cache(maxsize=1)
def _ddragon_versions() -> list:
    r = _requests.get("https://ddragon.leagueoflegends.com/api/versions.json",
                      headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=1)
def _ddragon_champions() -> dict:
    version = _ddragon_versions()[0]
    r = _requests.get(
        f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json",
        headers={"User-Agent": USER_AGENT}, timeout=15,
    )
    r.raise_for_status()
    payload = r.json()
    return {
        "version": version,
        "champions": sorted([
            {
                "id":   c["id"],
                "name": c["name"],
                "icon_url": f"https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{c['id']}.png",
            }
            for c in payload["data"].values()
        ], key=lambda c: c["name"]),
    }


@app.get("/champions")
def champions():
    """All current champions with names + icons (cached). Used by frontend autocomplete."""
    return _ddragon_champions()


@app.get("/champion-matchup")
def champion_matchup(champion1: str, champion2: str, role: Optional[str] = None):
    """Head-to-head WR for two champions facing each other (any role by default).

    Returns champion1's WR in games where champion2 was on the opposing team.
    `role` filters to picks of `champion1` in that role (so `Jhin` Bot vs `Ezreal`
    excludes off-role appearances).
    """
    key = f"cm:{champion1}|{champion2}|{role or ''}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    win_expr = (
        "CASE WHEN (pg.side = 'Blue' AND m.team1_side = 'Blue' AND m.winner = 1) "
        "       OR (pg.side = 'Blue' AND m.team2_side = 'Blue' AND m.winner = 2) "
        "       OR (pg.side = 'Red'  AND m.team1_side = 'Red'  AND m.winner = 1) "
        "       OR (pg.side = 'Red'  AND m.team2_side = 'Red'  AND m.winner = 2) "
        "  THEN 1 ELSE 0 END"
    )
    sql = f"""
        SELECT COUNT(*) AS g, SUM({win_expr}) AS w
        FROM player_games pg
        JOIN matches m ON m.game_id = pg.game_id
        JOIN player_games opp ON opp.game_id = pg.game_id
                             AND opp.side != pg.side
        WHERE pg.champion = ? AND opp.champion = ?
    """
    params: list = [champion1, champion2]
    if role:
        sql += " AND pg.role = ?"
        params.append(role)
    with get_conn() as conn:
        row = conn.execute(sql, params).fetchone()
    g = row["g"] or 0
    w = row["w"] or 0
    payload = {
        "champion1": champion1,
        "champion2": champion2,
        "role": role,
        "games":   g,
        "wins":    w,
        "winrate": (w / g) if g else None,
    }
    return _cache_put(key, payload)


@app.get("/champion-stats")
def champion_stats(
    champion: str,
    player: Optional[str] = None,
    role: Optional[str] = None,
):
    """Win-rate stats for a champion in the patch window.

    Returns the champion's overall WR plus, if `player` is provided, that
    player's record on this champion and their broader form (in `role` if
    given, else across all their games). `win` on player_games is unreliable
    in our ingestion, so we derive it by joining with `matches` on side.
    """
    key = f"champstats:{champion}|{player or ''}|{role or ''}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    win_expr = (
        "CASE WHEN (pg.side = 'Blue' AND m.team1_side = 'Blue' AND m.winner = 1) "
        "       OR (pg.side = 'Blue' AND m.team2_side = 'Blue' AND m.winner = 2) "
        "       OR (pg.side = 'Red'  AND m.team1_side = 'Red'  AND m.winner = 1) "
        "       OR (pg.side = 'Red'  AND m.team2_side = 'Red'  AND m.winner = 2) "
        "  THEN 1 ELSE 0 END"
    )

    payload: dict = {"champion": champion}
    with get_conn() as conn:
        # Global champion stats
        row = conn.execute(
            f"""
            SELECT COUNT(*) AS games, SUM({win_expr}) AS wins,
                   AVG(pg.kills) AS k, AVG(pg.deaths) AS d, AVG(pg.assists) AS a
            FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
            WHERE pg.champion = ?
            """,
            (champion,),
        ).fetchone()
        g = row["games"] or 0
        w = row["wins"] or 0
        payload["global"] = {
            "games":   g,
            "wins":    w,
            "winrate": (w / g) if g else None,
            "avg_kda": ((row["k"] or 0) + (row["a"] or 0)) / max(row["d"] or 1, 1),
        }

        if player:
            # Player on this champion
            pc = conn.execute(
                f"""
                SELECT COUNT(*) AS games, SUM({win_expr}) AS wins,
                       AVG(pg.kills) AS k, AVG(pg.deaths) AS d, AVG(pg.assists) AS a
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = ? AND pg.champion = ?
                """,
                (player, champion),
            ).fetchone()
            pg_games = pc["games"] or 0
            pg_wins  = pc["wins"] or 0
            payload["player_on_champion"] = {
                "player":  player,
                "games":   pg_games,
                "wins":    pg_wins,
                "winrate": (pg_wins / pg_games) if pg_games else None,
                "avg_kda": ((pc["k"] or 0) + (pc["a"] or 0)) / max(pc["d"] or 1, 1) if pg_games else None,
            }

            # Player overall (optionally scoped to role)
            sql = f"""
                SELECT COUNT(*) AS games, SUM({win_expr}) AS wins
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = ?
            """
            params: list = [player]
            if role:
                sql += " AND pg.role = ?"
                params.append(role)
            po = conn.execute(sql, params).fetchone()
            po_g = po["games"] or 0
            po_w = po["wins"] or 0
            payload["player_overall"] = {
                "player":  player,
                "role":    role,
                "games":   po_g,
                "wins":    po_w,
                "winrate": (po_w / po_g) if po_g else None,
            }

            # Player's most-played champs in this role (top 5) for context
            top_sql = f"""
                SELECT pg.champion, COUNT(*) AS games, SUM({win_expr}) AS wins
                FROM player_games pg JOIN matches m ON m.game_id = pg.game_id
                WHERE pg.player = ?
            """
            top_params: list = [player]
            if role:
                top_sql += " AND pg.role = ?"
                top_params.append(role)
            top_sql += " GROUP BY pg.champion ORDER BY games DESC LIMIT 5"
            top = conn.execute(top_sql, top_params).fetchall()
            payload["player_top_champs"] = [
                {
                    "champion": r["champion"],
                    "games":    r["games"],
                    "wins":     r["wins"] or 0,
                    "winrate":  (r["wins"] or 0) / r["games"] if r["games"] else None,
                }
                for r in top
            ]

    return _cache_put(key, payload)


@app.get("/team/{team}/champion-pool")
def team_champion_pool(team: str, role: Optional[str] = None):
    """Most-played champions for the team, optionally filtered by role.

    Useful for populating a champion picker once the lineup is known.
    """
    sql = """
        SELECT pg.role, pg.player, pg.champion, COUNT(*) AS games, AVG(pg.win) AS winrate
        FROM player_games pg
        WHERE pg.team = ? AND pg.champion IS NOT NULL
    """
    params: list = [team]
    if role:
        sql += " AND pg.role = ?"
        params.append(role)
    sql += " GROUP BY pg.role, pg.player, pg.champion ORDER BY games DESC"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


@app.get("/matches/recent")
def recent_matches(team: Optional[str] = None, limit: int = 20):
    sql = """
        SELECT game_id, datetime_utc, league_code, patch, team1, team2,
               team1_side, team2_side, winner
        FROM matches
    """
    params: list = []
    if team:
        sql += " WHERE team1 = ? OR team2 = ?"
        params += [team, team]
    sql += " ORDER BY datetime_utc DESC LIMIT ?"
    params.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _model_features_from_live(team1: str, team2: str, team1_side: str, feats: dict) -> dict:
    """Map the live-feature dict (from features/build) onto the training-feature
    names that the trained model expects. Live features overlap heavily with
    training features but use slightly different names; we translate here so
    the model pipeline can be reused without re-running feature engineering.
    """
    side_wr_t1 = feats.get("team1_blue_wr") if team1_side == "Blue" else feats.get("team1_red_wr")
    side_wr_t2 = feats.get("team2_red_wr")  if team1_side == "Blue" else feats.get("team2_blue_wr")
    t1_pchamp = feats.get("team1_player_champ_wr", 0.5)
    t2_pchamp = feats.get("team2_player_champ_wr", 0.5)
    t1_cg     = feats.get("team1_champ_global_wr", 0.5)
    t2_cg     = feats.get("team2_champ_global_wr", 0.5)
    t1_cl     = feats.get("team1_champ_league_wr", 0.5)
    t2_cl     = feats.get("team2_champ_league_wr", 0.5)
    return {
        "team1_games":     feats.get("team1_games", 0.0),
        "team1_winrate":   feats.get("team1_winrate", 0.0),
        "team1_avg_len":   feats.get("team1_avg_len", 0.0),
        "team2_games":     feats.get("team2_games", 0.0),
        "team2_winrate":   feats.get("team2_winrate", 0.0),
        "team2_avg_len":   feats.get("team2_avg_len", 0.0),
        "team1_side_wr":   side_wr_t1 or 0.0,
        "team2_side_wr":   side_wr_t2 or 0.0,
        "team1_side_blue": 1.0 if team1_side == "Blue" else 0.0,
        "h2h_games":       feats.get("h2h_games", 0.0),
        "h2h_team1_wr":    feats.get("h2h_team1_wr", 0.5),
        "team1_p_games":   0.0,
        "team1_p_winrate": feats.get("team1_player_wr_avg", 0.0),
        "team1_p_kda":     feats.get("team1_player_kda_avg", 0.0),
        "team2_p_games":   0.0,
        "team2_p_winrate": feats.get("team2_player_wr_avg", 0.0),
        "team2_p_kda":     feats.get("team2_player_kda_avg", 0.0),
        "team1_champ_global_wr": t1_cg,
        "team2_champ_global_wr": t2_cg,
        "team1_champ_league_wr": t1_cl,
        "team2_champ_league_wr": t2_cl,
        "team1_pchamp_wr":       t1_pchamp,
        "team2_pchamp_wr":       t2_pchamp,
        "champ_matchup_wr":      feats.get("champ_matchup_wr", 0.5),
        "wr_diff":         feats.get("wr_diff", 0.0),
        "p_wr_diff":       feats.get("pwr_diff", 0.0),
        "side_wr_diff":    (side_wr_t1 or 0.0) - (side_wr_t2 or 0.0),
        "champ_global_diff": t1_cg - t2_cg,
        "champ_league_diff": t1_cl - t2_cl,
        "pchamp_diff":       t1_pchamp - t2_pchamp,
    }


# Feature → group mapping for the prediction explanation. Anything in
# `FEATURE_COLS` not listed here falls into "other" (currently empty).
_FEATURE_GROUPS: dict[str, list[str]] = {
    "team_form":         ["team1_winrate", "team2_winrate", "team1_games", "team2_games",
                          "team1_avg_len", "team2_avg_len", "wr_diff"],
    "player_form":       ["team1_p_winrate", "team2_p_winrate", "team1_p_games", "team2_p_games",
                          "team1_p_kda", "team2_p_kda", "p_wr_diff"],
    "champion_picks":    ["team1_champ_global_wr", "team2_champ_global_wr", "champ_global_diff",
                          "team1_champ_league_wr", "team2_champ_league_wr", "champ_league_diff"],
    "player_on_champion":["team1_pchamp_wr", "team2_pchamp_wr", "pchamp_diff"],
    "champion_matchup":  ["champ_matchup_wr"],
    "head_to_head":      ["h2h_team1_wr", "h2h_games"],
    "side":              ["team1_side_wr", "team2_side_wr", "team1_side_blue", "side_wr_diff"],
}


def _explain_prediction(model, cols: list[str], mfeats: dict, full_p: float) -> dict:
    """Attribute the prediction to feature groups via "neutralization deltas".

    For each group, recompute the probability with that group's features held
    at the training mean (so they contribute zero to the standardized logit),
    then report `full_p - neutral_p`. Positive = the group helped team1.

    Logistic regression is non-linear in probability space (sigmoid), so the
    deltas don't sum exactly to (full_p - 0.5). They sum approximately, which
    is fine for explanation — the relative magnitudes are what matters.
    """
    scaler = model.named_steps["scaler"]
    means = list(scaler.mean_)
    col_to_idx = {c: i for i, c in enumerate(cols)}
    base_x = [mfeats[c] for c in cols]

    def proba(x_row):
        return float(model.predict_proba([x_row])[0][1])

    groups_out = []
    for group, feature_names in _FEATURE_GROUPS.items():
        x_neutral = list(base_x)
        active = []
        for fname in feature_names:
            i = col_to_idx.get(fname)
            if i is None:
                continue
            x_neutral[i] = means[i]  # hold at training mean -> zero standardized contribution
            active.append(fname)
        if not active:
            continue
        p_without = proba(x_neutral)
        delta = full_p - p_without
        groups_out.append({
            "group": group,
            "delta_team1_prob": round(delta, 4),  # +ve favors team1, -ve favors team2
            "p_without_group":  round(p_without, 4),
        })

    # Sort by absolute impact, biggest first.
    groups_out.sort(key=lambda r: -abs(r["delta_team1_prob"]))
    return {"groups": groups_out}


def _per_lane_advantage(team1_picks, team2_picks, feats) -> list[dict]:
    """Per-role lane advantage. Returns two independent edges per role so the
    frontend can show them as separate bars:
      - player_edge:  pc1_wr - pc2_wr (None if either side lacks sample >= 2)
      - matchup_edge: cm_wr - 0.5  (None if matchup sample < 5)
    Plus the raw stats so chips can display percentages.
    """
    if not (team1_picks and team2_picks):
        return []
    by_role_t1 = {p.role: p for p in team1_picks if p.role and p.champion}
    by_role_t2 = {p.role: p for p in team2_picks if p.role and p.champion}
    out = []
    with get_conn() as conn:
        from backend.features.build import _player_champ, _champion_matchup
        for role in ("Top", "Jungle", "Mid", "Bot", "Support"):
            s1 = by_role_t1.get(role); s2 = by_role_t2.get(role)
            if not (s1 and s2):
                continue
            cm = _champion_matchup(conn, s1.champion, s2.champion)
            entry = {
                "role":       role,
                "team1_player":   s1.player or None,
                "team1_champion": s1.champion,
                "team2_player":   s2.player or None,
                "team2_champion": s2.champion,
                # Champion-vs-champion matchup
                "matchup_games":   int(cm["games"]),
                "matchup_team1_wr": round(cm["winrate"], 4) if cm["games"] else None,
                "matchup_edge":    round(cm["winrate"] - 0.5, 4) if cm["games"] >= 5 else None,
                # Player-on-champion (filled below when sample exists)
                "team1_player_champ_wr":    None,
                "team2_player_champ_wr":    None,
                "team1_player_champ_games": 0,
                "team2_player_champ_games": 0,
                "player_edge": None,
            }
            if s1.player and s2.player:
                pc1 = _player_champ(conn, s1.player, s1.champion)
                pc2 = _player_champ(conn, s2.player, s2.champion)
                entry["team1_player_champ_games"] = int(pc1["champ_games"])
                entry["team2_player_champ_games"] = int(pc2["champ_games"])
                if pc1["champ_games"] >= 1:
                    entry["team1_player_champ_wr"] = round(pc1["champ_winrate"], 4)
                if pc2["champ_games"] >= 1:
                    entry["team2_player_champ_wr"] = round(pc2["champ_winrate"], 4)
                if pc1["champ_games"] >= 2 and pc2["champ_games"] >= 2:
                    entry["player_edge"] = round(pc1["champ_winrate"] - pc2["champ_winrate"], 4)
            out.append(entry)
    return out


@app.post("/predict")
def predict(req: PredictRequest):
    with get_conn() as conn:
        for team in (req.team1, req.team2):
            row = conn.execute(
                "SELECT 1 FROM matches WHERE team1 = ? OR team2 = ? LIMIT 1",
                (team, team),
            ).fetchone()
            if not row:
                raise HTTPException(404, f"No competitive history for team: {team!r}")

    inp = MatchInput(
        team1=req.team1,
        team2=req.team2,
        patch=req.patch,
        team1_side=req.team1_side,
        league=req.league,
        team1_players=[PlayerSelection(**p.model_dump()) for p in (req.team1_players or [])],
        team2_players=[PlayerSelection(**p.model_dump()) for p in (req.team2_players or [])],
    )
    feats = build_features(inp)

    bundle = load_model()
    explanation = None
    if bundle is not None:
        model = bundle["model"]
        cols  = bundle["feature_cols"]
        mfeats = _model_features_from_live(req.team1, req.team2, req.team1_side, feats)
        x = [[mfeats[c] for c in cols]]
        p = float(model.predict_proba(x)[0][1])
        which = "logreg-v1"
        explanation = _explain_prediction(model, cols, mfeats, p)
    else:
        p = baseline_probability(feats)
        which = "heuristic-v0"

    lanes = _per_lane_advantage(inp.team1_players, inp.team2_players, feats)

    return {
        "team1": req.team1,
        "team2": req.team2,
        "team1_win_probability": round(p, 4),
        "team2_win_probability": round(1 - p, 4),
        "model": which,
        "league": feats.get("league_context") or None,
        "features": {k: (round(v, 4) if isinstance(v, (int, float)) else v) for k, v in feats.items()},
        "explanation": explanation,
        "lane_advantages": lanes,
    }


# Serve the static frontend when the directory is present alongside `backend/`.
# This is purely additive — local `open frontend/index.html` still works because
# that flow doesn't go through FastAPI at all. On Railway / any single-service
# deploy this lets the same process serve both API and UI from the same origin,
# so the frontend's `location.hostname` check resolves to "" and API calls go
# to the same host.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND_DIR.is_dir():
    @app.get("/")
    def _index():
        return FileResponse(_FRONTEND_DIR / "index.html")

    # Mount remaining static assets under /static so we never shadow API routes.
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")
