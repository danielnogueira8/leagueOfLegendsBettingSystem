"""SQLite schema and helpers.

The model is denormalized enough to be easy to query for features without
heavy joins, but normalized enough to keep ingestion idempotent.

Primary keys mirror Leaguepedia's natural keys where possible (GameId, MatchId,
OverviewPage) so re-running ingestion is an upsert, never a duplicate.
"""
from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from backend.config import DB_PATH

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tournaments (
    overview_page TEXT PRIMARY KEY,
    name          TEXT,
    league        TEXT,
    league_code   TEXT,
    tier          TEXT,
    region        TEXT,
    year          INTEGER,
    date_start    TEXT,
    date_end      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tournaments_league ON tournaments(league_code);

CREATE TABLE IF NOT EXISTS teams (
    name          TEXT PRIMARY KEY,
    short_name    TEXT,
    region        TEXT
);

CREATE TABLE IF NOT EXISTS players (
    name          TEXT PRIMARY KEY,
    team          TEXT,
    role          TEXT,
    country       TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    -- A "match" here is a single game (best-of slot). Leaguepedia calls this ScoreboardGame.
    game_id        TEXT PRIMARY KEY,
    match_id       TEXT,
    overview_page  TEXT,
    league_code    TEXT,
    tournament     TEXT,
    patch          TEXT,
    datetime_utc   TEXT,
    team1          TEXT,
    team2          TEXT,
    winner         INTEGER,   -- 1 or 2
    team1_side     TEXT,      -- 'Blue' or 'Red'
    team2_side     TEXT,
    gamelength     REAL,
    FOREIGN KEY (overview_page) REFERENCES tournaments(overview_page)
);
CREATE INDEX IF NOT EXISTS idx_matches_patch  ON matches(patch);
CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_code);
CREATE INDEX IF NOT EXISTS idx_matches_team1  ON matches(team1);
CREATE INDEX IF NOT EXISTS idx_matches_team2  ON matches(team2);
CREATE INDEX IF NOT EXISTS idx_matches_dt     ON matches(datetime_utc);

CREATE TABLE IF NOT EXISTS player_games (
    -- One row per player per game.
    game_id        TEXT,
    player         TEXT,
    team           TEXT,
    side           TEXT,       -- 'Blue' or 'Red'
    role           TEXT,       -- Top/Jungle/Mid/Bot/Support
    champion       TEXT,
    kills          INTEGER,
    deaths         INTEGER,
    assists        INTEGER,
    cs             INTEGER,
    gold           INTEGER,
    win            INTEGER,    -- 0 or 1
    PRIMARY KEY (game_id, player),
    FOREIGN KEY (game_id) REFERENCES matches(game_id)
);
CREATE INDEX IF NOT EXISTS idx_pg_player    ON player_games(player);
CREATE INDEX IF NOT EXISTS idx_pg_champion  ON player_games(champion);
CREATE INDEX IF NOT EXISTS idx_pg_team      ON player_games(team);
CREATE INDEX IF NOT EXISTS idx_pg_role      ON player_games(role);
CREATE INDEX IF NOT EXISTS idx_pg_team_role     ON player_games(team, role);
CREATE INDEX IF NOT EXISTS idx_pg_player_champ  ON player_games(player, champion);
CREATE INDEX IF NOT EXISTS idx_matches_team1_dt ON matches(team1, datetime_utc);
CREATE INDEX IF NOT EXISTS idx_matches_team2_dt ON matches(team2, datetime_utc);

CREATE TABLE IF NOT EXISTS picks_bans (
    -- Optional: pick/ban order if we can pull it. Not required for v1 features.
    game_id        TEXT,
    team           TEXT,
    type           TEXT,       -- 'pick' or 'ban'
    champion       TEXT,
    sequence       INTEGER,
    PRIMARY KEY (game_id, team, type, sequence),
    FOREIGN KEY (game_id) REFERENCES matches(game_id)
);

CREATE TABLE IF NOT EXISTS patches (
    -- Snapshot of patches we care about. Refreshed each ingestion run.
    patch          TEXT PRIMARY KEY,
    is_current     INTEGER,    -- 1 if currently the live patch
    in_window      INTEGER,    -- 1 if within PATCH_WINDOW
    fetched_at     TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    -- Audit log of ingestion runs. Useful for debugging and incremental updates.
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at     TEXT,
    finished_at    TEXT,
    patches        TEXT,
    leagues        TEXT,
    matches_added  INTEGER,
    matches_seen   INTEGER,
    error          TEXT
);
"""


def init_db(db_path: Path | str = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()


@contextmanager
def get_conn(db_path: Path | str = DB_PATH):
    conn = sqlite3.connect(db_path, isolation_level="DEFERRED")
    conn.row_factory = sqlite3.Row
    # WAL gives concurrent readers + a single writer (good for API + ingestion
    # running side-by-side). NORMAL sync is durable enough for an analytics DB.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=134217728")  # 128 MiB
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Initialized SQLite DB at {DB_PATH}")
