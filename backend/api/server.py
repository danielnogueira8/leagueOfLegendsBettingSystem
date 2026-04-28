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
from pydantic import BaseModel

import json
import time
from functools import lru_cache
from pathlib import Path

import requests as _requests

from backend.config import USER_AGENT, DATA_DIR
from backend.db.schema import get_conn
from backend.features.build import (
    MatchInput, PlayerSelection, build_features, baseline_probability,
)
from backend.models.train import load_model
from backend.models.dataset import FEATURE_COLS

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
        "wr_diff":         feats.get("wr_diff", 0.0),
        "p_wr_diff":       feats.get("pwr_diff", 0.0),
        "side_wr_diff":    (side_wr_t1 or 0.0) - (side_wr_t2 or 0.0),
    }


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
        team1_players=[PlayerSelection(**p.model_dump()) for p in (req.team1_players or [])],
        team2_players=[PlayerSelection(**p.model_dump()) for p in (req.team2_players or [])],
    )
    feats = build_features(inp)

    bundle = load_model()
    if bundle is not None:
        model = bundle["model"]
        cols  = bundle["feature_cols"]
        mfeats = _model_features_from_live(req.team1, req.team2, req.team1_side, feats)
        x = [[mfeats[c] for c in cols]]
        p = float(model.predict_proba(x)[0][1])
        which = "logreg-v1"
    else:
        p = baseline_probability(feats)
        which = "heuristic-v0"

    return {
        "team1": req.team1,
        "team2": req.team2,
        "team1_win_probability": round(p, 4),
        "team2_win_probability": round(1 - p, 4),
        "model": which,
        "features": {k: round(v, 4) for k, v in feats.items()},
    }
