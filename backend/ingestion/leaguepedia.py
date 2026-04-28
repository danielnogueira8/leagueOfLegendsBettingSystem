"""Leaguepedia (lol.fandom.com) Cargo API client.

Cargo is MediaWiki's structured-data extension; Leaguepedia exposes pro match
data through it.

Anonymous Cargo queries get aggressively rate-limited because Cargo executes
raw SQL. We log in with a Fandom account (credentials in .env) which gives a
much higher rate-limit ceiling and uses a session cookie via requests.Session.

Docs: https://lol.fandom.com/wiki/Help:Cargo

Key tables we use:
  - ScoreboardGames:   one row per pro game, with patch/teams/winner/etc.
  - ScoreboardPlayers: one row per player per game, with champion/role/KDA.
  - PicksAndBansS7:    pick/ban order (optional; v1 doesn't need it).
  - Tournaments:       tournament metadata (league, date range, tier).

Pagination: Cargo caps results at 500 per call; we page with offset.
"""
from __future__ import annotations
import os
import requests
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from backend.config import USER_AGENT, PROJECT_ROOT

API_URL = "https://lol.fandom.com/api.php"
PAGE_LIMIT = 500
TIMEOUT = 30
MIN_INTERVAL = 1.1     # seconds between requests, even when authenticated
RETRY_SLEEP = 10.0
_last_request_at: float = 0.0
_session: Optional[requests.Session] = None


def _load_env() -> None:
    """Tiny .env loader so we don't need python-dotenv as a dep."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _login() -> requests.Session:
    """Log into Fandom via the MediaWiki API.

    Two-step login: fetch a login token, then POST credentials with that token.
    The session keeps the auth cookie for subsequent calls.
    """
    _load_env()
    user = os.environ.get("LEAGUEPEDIA_USERNAME")
    pw = os.environ.get("LEAGUEPEDIA_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            "LEAGUEPEDIA_USERNAME / LEAGUEPEDIA_PASSWORD not set in .env"
        )

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})

    # 1) Get login token.
    r = s.get(
        API_URL,
        params={"action": "query", "meta": "tokens", "type": "login", "format": "json"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    token = r.json()["query"]["tokens"]["logintoken"]

    # 2) Submit credentials via the legacy `action=login` endpoint.
    # Fandom's modern `clientlogin` rejects raw passwords for security; the
    # legacy endpoint still works for regular Fandom accounts and is also the
    # accepted path for Special:BotPasswords (username form "User@BotName").
    r = s.post(
        API_URL,
        data={
            "action": "login",
            "lgname": user,
            "lgpassword": pw,
            "lgtoken": token,
            "format": "json",
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    payload = r.json().get("login", {})
    if payload.get("result") != "Success":
        raise RuntimeError(f"Leaguepedia login failed: {payload}")
    return s


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = _login()
    return _session


def _throttle() -> None:
    global _last_request_at
    elapsed = time.monotonic() - _last_request_at
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request_at = time.monotonic()


def _get(params: Dict[str, Any], retries: int = 5) -> Dict[str, Any]:
    params = {**params, "format": "json"}
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            _throttle()
            session = _get_session()
            r = session.get(API_URL, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                err = data["error"]
                if isinstance(err, dict) and err.get("code") == "ratelimited":
                    time.sleep(RETRY_SLEEP * (2 ** attempt))
                    last_err = RuntimeError(f"ratelimited: {err.get('info')}")
                    continue
                raise RuntimeError(f"Leaguepedia API error: {err}")
            return data
        except (requests.RequestException, ValueError, RuntimeError) as e:
            last_err = e
            time.sleep(RETRY_SLEEP * (attempt + 1))
    raise RuntimeError(f"Leaguepedia request failed after {retries} retries: {last_err}")


def cargo_query(
    tables: str,
    fields: str,
    where: Optional[str] = None,
    join_on: Optional[str] = None,
    order_by: Optional[str] = None,
    group_by: Optional[str] = None,
) -> Iterator[Dict[str, Any]]:
    """Stream all rows from a Cargo query, paginating transparently."""
    offset = 0
    while True:
        params: Dict[str, Any] = {
            "action": "cargoquery",
            "tables": tables,
            "fields": fields,
            "limit": PAGE_LIMIT,
            "offset": offset,
        }
        if where:    params["where"]    = where
        if join_on:  params["join_on"]  = join_on
        if order_by: params["order_by"] = order_by
        if group_by: params["group_by"] = group_by

        data = _get(params)
        rows = data.get("cargoquery", [])
        if not rows:
            return
        for row in rows:
            yield row["title"]  # Cargo wraps each row in {'title': {...}}
        if len(rows) < PAGE_LIMIT:
            return
        offset += PAGE_LIMIT


# ---------- Domain queries ---------- #

# Leaguepedia stores patches as strings like "16.08" or "16.8" depending on era.
# We normalize on the way in. Some older rows have "14.21" already normalized.
def _norm_patch(p: Optional[str]) -> Optional[str]:
    if not p:
        return None
    parts = p.strip().split(".")
    if len(parts) >= 2:
        try:
            return f"{int(parts[0])}.{int(parts[1])}"
        except ValueError:
            return p
    return p


def fetch_tournament_overviewpages(leagues: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield Tournaments rows whose League is in `leagues`.

    Used as the first step of game ingestion: get all OverviewPages for the
    leagues we care about, then query games by OverviewPage. This avoids a
    multi-table JOIN that's currently triggering MWException on Cargo.
    """
    if not leagues:
        return
    league_in = ",".join(f'"{l}"' for l in leagues)
    fields = ",".join([
        "T.OverviewPage=OverviewPage",
        "T.Name=Name",
        "T.League=League",
        "T.Region=Region",
        "T.Year=Year",
        "T.DateStart=DateStart",
        "T.Date=DateEnd",
    ])
    yield from cargo_query(
        tables="Tournaments=T",
        fields=fields,
        where=f"T.League IN ({league_in})",
    )


def fetch_games_for_patches(patches: List[str], leagues: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield ScoreboardGames rows for the given patches and leagues.

    Two-step strategy (avoids Cargo's JOIN-with-ORDER-BY MWException bug):
      1) Find OverviewPages of tournaments in the desired leagues.
      2) Query ScoreboardGames with `Patch IN ... AND OverviewPage IN ...`.

    Yields rows that already include League/Region/Year (joined in client-side
    from the tournaments lookup) so the orchestrator can keep its current
    column expectations.
    """
    if not patches or not leagues:
        return

    # Step 1: tournaments in scope.
    tournaments: Dict[str, Dict[str, Any]] = {}
    for trow in fetch_tournament_overviewpages(leagues):
        op = trow.get("OverviewPage")
        if op:
            tournaments[op] = trow

    if not tournaments:
        return

    patch_in = ",".join(f'"{p}"' for p in patches)

    # Note: ScoreboardGames does not expose Team1Side/Team2Side — Cargo throws
    # MWException if you select them. Side info is derived later from
    # ScoreboardPlayers.Side (1 = Blue, 2 = Red).
    fields = ",".join([
        "SG.GameId=GameId",
        "SG.OverviewPage=OverviewPage",
        "SG.Tournament=Tournament",
        "SG.Patch=Patch",
        "SG.DateTime_UTC=DateTimeUTC",
        "SG.Team1=Team1",
        "SG.Team2=Team2",
        "SG.Winner=Winner",
        "SG.Gamelength_Number=Gamelength",
    ])

    # Cargo struggles with very large IN lists. Chunk OverviewPages to stay safe.
    CHUNK = 40
    op_keys = list(tournaments.keys())
    for i in range(0, len(op_keys), CHUNK):
        chunk = op_keys[i:i + CHUNK]
        ops = ",".join(f'"{o}"' for o in chunk)
        where = f'SG.Patch IN ({patch_in}) AND SG.OverviewPage IN ({ops})'
        for row in cargo_query(
            tables="ScoreboardGames=SG",
            fields=fields,
            where=where,
        ):
            t = tournaments.get(row.get("OverviewPage"))
            if t:
                row["League"] = t.get("League")
                row["Region"] = t.get("Region")
                row["Year"] = t.get("Year")
            yield row


def fetch_player_games(game_ids: List[str]) -> Iterator[Dict[str, Any]]:
    """Yield ScoreboardPlayers rows for the given GameIds."""
    if not game_ids:
        return
    # Chunk to keep WHERE clause reasonable.
    CHUNK = 50
    fields = ",".join([
        "SP.GameId=GameId",
        "SP.Link=Player",
        "SP.Team=Team",
        "SP.Side=Side",
        "SP.Role=Role",
        "SP.Champion=Champion",
        "SP.Kills=Kills",
        "SP.Deaths=Deaths",
        "SP.Assists=Assists",
        "SP.CS=CS",
        "SP.Gold=Gold",
        "SP.PlayerWin=PlayerWin",
    ])
    for i in range(0, len(game_ids), CHUNK):
        chunk = game_ids[i:i + CHUNK]
        gids = ",".join(f'"{g}"' for g in chunk)
        where = f"SP.GameId IN ({gids})"
        yield from cargo_query(
            tables="ScoreboardPlayers=SP",
            fields=fields,
            where=where,
        )


def fetch_tournaments(overview_pages: List[str]) -> Iterator[Dict[str, Any]]:
    if not overview_pages:
        return
    CHUNK = 50
    fields = ",".join([
        "T.OverviewPage=OverviewPage",
        "T.Name=Name",
        "T.League=League",
        "T.Region=Region",
        "T.Year=Year",
        "T.DateStart=DateStart",
        "T.Date=DateEnd",
    ])
    for i in range(0, len(overview_pages), CHUNK):
        pages = overview_pages[i:i + CHUNK]
        ops = ",".join(f'"{p}"' for p in pages)
        where = f"T.OverviewPage IN ({ops})"
        yield from cargo_query(tables="Tournaments=T", fields=fields, where=where)


if __name__ == "__main__":
    # Smoke test: pull a couple games from a recent patch.
    from backend.ingestion.data_dragon import patch_window
    patches = patch_window(1)
    print(f"Probing patches: {patches}")
    n = 0
    for row in fetch_games_for_patches(patches, ["LCK", "LEC"]):
        print(row)
        n += 1
        if n >= 3:
            break
    print(f"OK, fetched {n} sample rows")
