"""Project configuration.

Single source of truth for league scope, patch window, and paths.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Allow override via env var so deploys can point at a mounted volume
# (Railway/Fly/etc.) without changing code. Defaults to ./data for local use.
DATA_DIR = Path(os.environ.get("LOL_DATA_DIR") or (PROJECT_ROOT / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "lol_betting.db"

# How many patches back to ingest (current + N-1 prior). 3 means current + 2 prior.
PATCH_WINDOW = 3

# Leaguepedia exposes leagues via two fields:
#   - Leagues.League_Short (e.g. "LCK")
#   - Tournaments.League   (the canonical full name, e.g. "LoL Champions Korea")
# Cargo's Tournaments table stores the canonical full name, so we filter on that.
# Keys below are our stable internal codes; `league_full` is what we filter on.
LEAGUES = {
    # Major regions
    "LCK":   {"league_full": "LoL Champions Korea",                            "tier": "major"},
    "LEC":   {"league_full": "LoL EMEA Championship",                          "tier": "major"},
    # NA: LCS still operates (was NOT replaced by LTA — LTA is a separate
    # tournament running in parallel for select teams across the Americas).
    "LCS":   {"league_full": "League of Legends Championship Series",                 "tier": "major"},
    "LTA":   {"league_full": "League of Legends Championship of The Americas",        "tier": "major"},
    "LTAN":  {"league_full": "League of Legends Championship of The Americas North",  "tier": "major"},
    "LTAS":  {"league_full": "League of Legends Championship of The Americas South",  "tier": "major"},
    "LPL":   {"league_full": "Tencent LoL Pro League",                         "tier": "major"},
    # International events
    "MSI":   {"league_full": "Mid-Season Invitational",                        "tier": "international"},
    "WCS":   {"league_full": "World Championship",                             "tier": "international"},
    "FST":   {"league_full": "First Stand",                                    "tier": "international"},
    "EWC":   {"league_full": "Esports World Cup",                              "tier": "international"},
    # Asia-Pacific (LCP replaced PCS/VCS in 2025; we keep both for backfill)
    "LCP":   {"league_full": "League of Legends Championship Pacific",         "tier": "major"},
    "PCS":   {"league_full": "Pacific Championship Series",                    "tier": "major"},
    "VCS":   {"league_full": "Vietnam Championship Series",                    "tier": "major"},
    # Brazil / LatAm / Japan
    "CBLOL": {"league_full": "Circuit Brazilian League of Legends",            "tier": "major"},
    "LLA":   {"league_full": "Liga Latinoamerica",                             "tier": "major"},
    "LJL":   {"league_full": "LoL Japan League",                               "tier": "major"},
    # European regional / academy
    "LFL":       {"league_full": "La Ligue Française",                         "tier": "regional"},
    "PRM":       {"league_full": "Prime League Pro Division",                  "tier": "regional"},
    "NLC":       {"league_full": "Northern League of Legends Championship",    "tier": "regional"},
    "UL":        {"league_full": "Ultraliga",                                  "tier": "regional"},
    "HM":        {"league_full": "Hitpoint Masters",                           "tier": "regional"},
    "SL":        {"league_full": "LVP SuperLiga",                              "tier": "regional"},
    "EM":        {"league_full": "EMEA Masters",                               "tier": "regional"},
    # Korea academy / Asia-Pacific wildcard
    "LCKCL":     {"league_full": "LCK Challengers League",                     "tier": "academy"},
    "LCPW":      {"league_full": "LCP Wildcard League",                        "tier": "academy"},
}

USER_AGENT = "LoLBettingSystem/0.1 (research; contact: danielhenriquesnogueira@gmail.com)"
