"""Ingestion orchestrator.

Pulls all pro games on the active patch window across configured leagues,
plus per-player game rows and tournament metadata, and upserts them into
SQLite. Idempotent: re-running this only adds new data and updates existing.

Run:
    python -m backend.ingestion.run
"""
from __future__ import annotations
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from backend.config import LEAGUES, PATCH_WINDOW
from backend.db.schema import init_db, get_conn
from backend.ingestion import data_dragon, leaguepedia


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_patch(p: Optional[str]) -> Optional[str]:
    """Pass through the patch string as Leaguepedia stores it (e.g. '26.08').

    Leaguepedia is the source of truth for the human-facing patch identifier;
    we don't re-normalize so the orchestrator + features stay aligned with the
    patch_window query.
    """
    return p.strip() if p else None


def _league_code_for(league_full: Optional[str]) -> Optional[str]:
    """Map Leaguepedia's canonical League name back to our internal code."""
    if not league_full:
        return None
    for code, cfg in LEAGUES.items():
        if cfg["league_full"].lower() == league_full.lower():
            return code
    return None


def _to_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def upsert_match(conn: sqlite3.Connection, row: Dict[str, Any]) -> bool:
    """Returns True if a new row was inserted, False if it was an update."""
    game_id = row.get("GameId")
    if not game_id:
        return False
    league_code = _league_code_for(row.get("League"))
    patch = _norm_patch(row.get("Patch"))

    cur = conn.execute("SELECT 1 FROM matches WHERE game_id = ?", (game_id,))
    existed = cur.fetchone() is not None

    # team1_side / team2_side are filled in after player ingestion via
    # backfill_sides(); here we just leave them NULL initially.
    conn.execute(
        """
        INSERT INTO matches (
            game_id, match_id, overview_page, league_code, tournament,
            patch, datetime_utc, team1, team2, winner, team1_side, team2_side,
            gamelength
        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            overview_page = excluded.overview_page,
            league_code   = excluded.league_code,
            tournament    = excluded.tournament,
            patch         = excluded.patch,
            datetime_utc  = excluded.datetime_utc,
            team1         = excluded.team1,
            team2         = excluded.team2,
            winner        = excluded.winner,
            gamelength    = excluded.gamelength
        """,
        (
            game_id,
            row.get("OverviewPage"),
            league_code,
            row.get("Tournament"),
            patch,
            row.get("DateTimeUTC"),
            row.get("Team1"),
            row.get("Team2"),
            _to_int(row.get("Winner")),
            _to_float(row.get("Gamelength")),
        ),
    )

    # Touch teams referenced by this match.
    for team in (row.get("Team1"), row.get("Team2")):
        if team:
            conn.execute(
                "INSERT OR IGNORE INTO teams (name, region) VALUES (?, ?)",
                (team, row.get("Region")),
            )

    return not existed


def _side_to_str(v: Any) -> Optional[str]:
    """Leaguepedia stores Side as integer (1=Blue, 2=Red). Normalize to text."""
    if v is None or v == "":
        return None
    try:
        i = int(v)
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        return "Blue" if s in ("1", "blue") else "Red" if s in ("2", "red") else None
    return "Blue" if i == 1 else "Red" if i == 2 else None


def upsert_player_game(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    game_id = row.get("GameId")
    player = row.get("Player")
    if not game_id or not player:
        return
    conn.execute(
        """
        INSERT INTO player_games (
            game_id, player, team, side, role, champion,
            kills, deaths, assists, cs, gold, win
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id, player) DO UPDATE SET
            team     = excluded.team,
            side     = excluded.side,
            role     = excluded.role,
            champion = excluded.champion,
            kills    = excluded.kills,
            deaths   = excluded.deaths,
            assists  = excluded.assists,
            cs       = excluded.cs,
            gold     = excluded.gold,
            win      = excluded.win
        """,
        (
            game_id,
            player,
            row.get("Team"),
            _side_to_str(row.get("Side")),
            row.get("Role"),
            row.get("Champion"),
            _to_int(row.get("Kills")),
            _to_int(row.get("Deaths")),
            _to_int(row.get("Assists")),
            _to_int(row.get("CS")),
            _to_int(row.get("Gold")),
            _to_int(row.get("PlayerWin")),
        ),
    )
    # Keep player roster table populated.
    conn.execute(
        """
        INSERT INTO players (name, team, role) VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            team = COALESCE(excluded.team, players.team),
            role = COALESCE(excluded.role, players.role)
        """,
        (player, row.get("Team"), row.get("Role")),
    )


def upsert_tournament(conn: sqlite3.Connection, row: Dict[str, Any]) -> None:
    op = row.get("OverviewPage")
    if not op:
        return
    league_code = _league_code_for(row.get("League"))
    tier = LEAGUES.get(league_code, {}).get("tier") if league_code else None
    conn.execute(
        """
        INSERT INTO tournaments (
            overview_page, name, league, league_code, tier, region, year, date_start, date_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(overview_page) DO UPDATE SET
            name        = excluded.name,
            league      = excluded.league,
            league_code = excluded.league_code,
            tier        = excluded.tier,
            region      = excluded.region,
            year        = excluded.year,
            date_start  = excluded.date_start,
            date_end    = excluded.date_end
        """,
        (
            op,
            row.get("Name"),
            row.get("League"),
            league_code,
            tier,
            row.get("Region"),
            _to_int(row.get("Year")),
            row.get("DateStart"),
            row.get("DateEnd"),
        ),
    )


def backfill_sides(conn: sqlite3.Connection) -> int:
    """Populate matches.team1_side and team2_side from per-player Side rows.

    For each match, look at any player_games row whose team matches matches.team1
    and use that row's side ('Blue' / 'Red'). Returns # matches updated.
    """
    updated = conn.execute(
        """
        WITH sides AS (
            SELECT
                m.game_id,
                MAX(CASE WHEN pg.team = m.team1 THEN pg.side END) AS s1,
                MAX(CASE WHEN pg.team = m.team2 THEN pg.side END) AS s2
            FROM matches m
            JOIN player_games pg ON pg.game_id = m.game_id
            WHERE (m.team1_side IS NULL OR m.team2_side IS NULL)
            GROUP BY m.game_id
        )
        UPDATE matches
           SET team1_side = (SELECT s1 FROM sides WHERE sides.game_id = matches.game_id),
               team2_side = (SELECT s2 FROM sides WHERE sides.game_id = matches.game_id)
         WHERE game_id IN (SELECT game_id FROM sides)
        """
    ).rowcount
    return updated


def refresh_patch_table(conn: sqlite3.Connection, patches: List[str], current: str) -> None:
    conn.execute("DELETE FROM patches")
    now = _utcnow_iso()
    for p in patches:
        conn.execute(
            "INSERT INTO patches (patch, is_current, in_window, fetched_at) VALUES (?, ?, ?, ?)",
            (p, 1 if p == current else 0, 1, now),
        )


def run_ingestion(patch_window: int = PATCH_WINDOW, league_codes: Optional[List[str]] = None) -> Dict[str, Any]:
    init_db()
    started = _utcnow_iso()

    current = data_dragon.current_patch()
    patches = data_dragon.patch_window(patch_window)

    leagues_cfg = LEAGUES if not league_codes else {k: v for k, v in LEAGUES.items() if k in league_codes}
    league_full_names = [v["league_full"] for v in leagues_cfg.values()]

    print(f"[ingest] current patch={current}  window={patches}", flush=True)
    print(f"[ingest] leagues ({len(league_full_names)}): {list(leagues_cfg.keys())}", flush=True)

    matches_added = 0
    matches_seen = 0
    error: Optional[str] = None
    overview_pages: set[str] = set()
    game_ids: List[str] = []

    try:
        with get_conn() as conn:
            refresh_patch_table(conn, patches, current)
            t0 = time.monotonic()
            for row in leaguepedia.fetch_games_for_patches(patches, league_full_names):
                matches_seen += 1
                if upsert_match(conn, row):
                    matches_added += 1
                if row.get("OverviewPage"):
                    overview_pages.add(row["OverviewPage"])
                if row.get("GameId"):
                    game_ids.append(row["GameId"])
                if matches_seen % 100 == 0:
                    print(f"[ingest] matches: seen={matches_seen} added={matches_added} elapsed={time.monotonic()-t0:.1f}s", flush=True)
            print(f"[ingest] matches done: seen={matches_seen} added={matches_added}", flush=True)

            print(f"[ingest] fetching tournament metadata for {len(overview_pages)} tournaments…", flush=True)
            for trow in leaguepedia.fetch_tournaments(sorted(overview_pages)):
                upsert_tournament(conn, trow)

            print(f"[ingest] fetching player rows for {len(game_ids)} games…", flush=True)
            n_pg = 0
            for prow in leaguepedia.fetch_player_games(game_ids):
                upsert_player_game(conn, prow)
                n_pg += 1
                if n_pg % 1000 == 0:
                    print(f"[ingest]   player rows: {n_pg}", flush=True)
            print(f"[ingest] player rows done: {n_pg}", flush=True)

            print("[ingest] backfilling team sides…", flush=True)
            n_sides = backfill_sides(conn)
            print(f"[ingest] sides backfilled for {n_sides} matches", flush=True)
    except Exception as e:
        error = repr(e)
        print(f"[ingest] ERROR: {error}", flush=True)

    finished = _utcnow_iso()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO ingestion_runs (
                started_at, finished_at, patches, leagues, matches_added, matches_seen, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started, finished,
                json.dumps(patches),
                json.dumps(list(leagues_cfg.keys())),
                matches_added, matches_seen, error,
            ),
        )

    return {
        "started": started,
        "finished": finished,
        "patches": patches,
        "leagues": list(leagues_cfg.keys()),
        "matches_added": matches_added,
        "matches_seen": matches_seen,
        "error": error,
    }


if __name__ == "__main__":
    # Allow filtering: `python -m backend.ingestion.run LCK LEC`
    league_filter = sys.argv[1:] or None
    summary = run_ingestion(league_codes=league_filter)
    print("\n[ingest] summary:")
    print(json.dumps(summary, indent=2))
