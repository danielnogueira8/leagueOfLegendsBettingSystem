# League of Legends Match Predictor

Win-probability predictions for upcoming pro-League-of-Legends matches, trained
**only on competitive games** from the most recent patches. No solo queue data.

## What it does

- Pulls every pro game from [Leaguepedia](https://lol.fandom.com/) across 22+
  major and regional leagues for the **current patch and the two prior patches**
  (a rolling window — older data is automatically excluded).
- Builds team-form, side-winrate, head-to-head, player-form, and player-on-champion
  features.
- Trains a logistic-regression model with a leak-safe time-based train/val split.
- Serves predictions through a FastAPI endpoint and a small frontend.

## Stack

- Python 3.11+, SQLite, FastAPI, scikit-learn
- Static HTML/JS frontend (no build step)

## Setup

```bash
pip install -r requirements.txt
```

Create `.env` in the project root with Leaguepedia bot-password credentials
(needed because anonymous Cargo queries get rate-limited aggressively):

```env
LEAGUEPEDIA_USERNAME=YourFandomUser@BotName
LEAGUEPEDIA_PASSWORD=long-bot-password-from-Special:BotPasswords
```

Get a bot password at <https://lol.fandom.com/wiki/Special:BotPasswords>
(grants needed: "Basic rights" + "High-volume editing").

## Usage

### One-shot refresh (ingest + retrain)

```bash
python -m backend.refresh
```

Takes ~10 minutes for the full ingestion across all leagues + ~2 seconds to
retrain. Idempotent — safe to re-run.

### Run components separately

```bash
# Ingest just specific leagues
python -m backend.ingestion.run LCK LEC LPL

# Train model from whatever's in the DB
python -m backend.models.train
```

### Start the API

```bash
uvicorn backend.api.server:app --reload --port 8000
```

Then open `frontend/index.html` in a browser. The page hits
`http://localhost:8000` directly.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | DB stats + readiness |
| GET  | `/patches` | Patches in scope, current marked |
| GET  | `/teams?league=LCK` | Teams seen in window |
| GET  | `/players?team=T1` | Players known for a team |
| GET  | `/team/{team}/stats` | Win counts, side WR, recent matches |
| GET  | `/team/{team}/lineup` | Auto-detected starting lineup |
| GET  | `/team/{team}/champion-pool` | Most-played champs per role |
| GET  | `/h2h?team1=A&team2=B` | Head-to-head record + recent meetings |
| GET  | `/champions` | All current champions (icons + names) |
| GET  | `/matches/recent?team=T1` | Recent games involving a team |
| POST | `/predict` | Win probability for a matchup |

### `/predict` request

```json
{
  "team1": "T1",
  "team2": "Gen.G",
  "team1_side": "Blue",
  "team1_players": [
    {"player": "Faker", "champion": "Azir", "role": "Mid"}
  ]
}
```

`team1_players` / `team2_players` are optional — omit for a team-level
prediction without champion-specific signal.

## Project layout

```
backend/
  config.py               # League scope, patch window, paths
  db/schema.py            # SQLite schema + connection helper
  ingestion/
    leaguepedia.py        # Cargo client (auth + throttle)
    data_dragon.py        # Patch detection
    run.py                # Orchestrator (idempotent upserts)
  features/build.py       # Live feature engineering for /predict
  models/
    dataset.py            # Training-row builder (point-in-time correct)
    train.py              # Logistic regression + metrics
  api/server.py           # FastAPI app
  refresh.py              # One-shot ingest + retrain
frontend/
  index.html              # Static UI
data/                     # SQLite DB + trained model (gitignored)
```

## Notes

- **Patch detection is dynamic.** Reads the latest patch from Leaguepedia,
  then takes that patch + 2 prior. When a new patch lands, the next refresh
  drops the now-out-of-window patch automatically.
- **Idempotent ingestion.** All upserts key on Leaguepedia's natural IDs
  (`GameId`, `OverviewPage`), so re-running is safe.
- **Leak-safe training.** Every training row uses only matches that pre-date
  its target. The train/val split is time-based — last 20% of the timeline
  is validation.
- **Calibration.** Logistic regression gives well-calibrated probabilities out
  of the box. If we move to gradient boosting later, we'll need post-hoc
  calibration (sigmoid or isotonic).
