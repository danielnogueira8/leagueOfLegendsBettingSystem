"""Data Dragon client.

Used to determine the current live patch and to translate "match patch"
strings (e.g. "16.8") into a normalized form. Data Dragon versions look like
"14.21.1" (major.minor.hotfix). The "patch" we care about is "14.21".
"""
from __future__ import annotations
import requests
from functools import lru_cache
from typing import List

from backend.config import USER_AGENT

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
TIMEOUT = 15


def _short(version: str) -> str:
    """'14.21.1' -> '14.21'. Non-numeric versions returned unchanged."""
    parts = version.split(".")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return version


@lru_cache(maxsize=1)
def fetch_versions() -> List[str]:
    r = requests.get(VERSIONS_URL, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def current_engine_patch() -> str:
    """Data Dragon's latest engine version (e.g. '16.8'). Use for static assets."""
    return _short(fetch_versions()[0])


def current_patch() -> str:
    """Patch identifier as used on Leaguepedia / casters / fans (e.g. '26.08').

    Leaguepedia stores `ScoreboardGames.Patch` in the human-visible form
    (season-aligned, "26.08" for season 2026 patch 8). That is what we need
    to match in Cargo queries. Data Dragon's engine version (16.8) does not
    appear in Leaguepedia data.
    """
    # Imported here to avoid a circular import at module-load time.
    from backend.ingestion.leaguepedia import _get
    data = _get({
        "action": "cargoquery",
        "tables": "ScoreboardGames=SG",
        "fields": "SG.Patch=Patch",
        "where": "SG.Patch IS NOT NULL",
        "order_by": "SG.DateTime_UTC DESC",
        "limit": 1,
    })
    rows = data.get("cargoquery", [])
    if not rows:
        raise RuntimeError("Could not determine current patch from Leaguepedia")
    return rows[0]["title"]["Patch"].strip()


def patch_window(window: int) -> List[str]:
    """The last `window` distinct patches as recorded on Leaguepedia, newest first.

    Strategy: query distinct patches via Cargo's `group_by`, ordered by the
    max timestamp on each patch, descending. This guarantees we see the latest
    N regardless of how many hotfix rows exist on the most recent patch.
    """
    from backend.ingestion.leaguepedia import _get
    data = _get({
        "action": "cargoquery",
        "tables": "ScoreboardGames=SG",
        "fields": "SG.Patch=Patch, MAX(SG.DateTime_UTC)=MaxDate",
        "where": "SG.Patch IS NOT NULL",
        "group_by": "SG.Patch",
        "order_by": "MAX(SG.DateTime_UTC) DESC",
        "limit": max(window * 5, 20),
    })
    seen: list[str] = []
    for row in data.get("cargoquery", []):
        p = (row["title"].get("Patch") or "").strip()
        if not p:
            continue
        if p not in seen:
            seen.append(p)
        if len(seen) >= window:
            break
    return seen


if __name__ == "__main__":
    print("Engine patch (Data Dragon):", current_engine_patch())
    print("Current patch (Leaguepedia):", current_patch())
    print("Last 3 patches (Leaguepedia):", patch_window(3))
