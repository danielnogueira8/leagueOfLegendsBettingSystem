# Deploying to Railway

The backend serves both the API and the frontend in a single service.

## One-time setup

1. **Push this repo to GitHub** (Railway pulls from there).

2. **Create a Railway project** → New Project → Deploy from GitHub repo →
   pick this repo. Railway auto-detects Python via Nixpacks and uses the
   `Procfile` / `railway.toml` in the repo root.

3. **Add a persistent volume** (Settings → Volumes):
   - Mount path: `/app/data`
   - Size: 1 GB is plenty
   This is where the SQLite DB and trained model live across deploys.

4. **Set environment variables** (Settings → Variables):
   ```
   LOL_DATA_DIR=/app/data
   LEAGUEPEDIA_USERNAME=YourFandomUser@BotName
   LEAGUEPEDIA_PASSWORD=long-bot-password
   ```
   Get a bot password at https://lol.fandom.com/wiki/Special:BotPasswords
   (grants needed: "Basic rights" + "High-volume editing").

5. **First deploy** will boot with an empty DB (`/health` returns 0 matches).
   Open Railway's shell tab and run:
   ```
   python -m backend.refresh
   ```
   ~10 minutes. After it finishes, `/health` shows real numbers and the
   frontend works.

## Visiting the app

The Railway URL serves both:
- `https://<app>.railway.app/` — frontend
- `https://<app>.railway.app/health`, `/predict`, etc — API

The frontend autodetects the host, so no config needed.

## Keeping data fresh

Set up a Railway cron job (Settings → Cron) to re-ingest periodically:
- Schedule: `0 12 * * *` (daily at noon UTC) — fine for tracking new patches
- Command: `python -m backend.refresh`

## Local development is unchanged

Nothing in the local flow changed. You can still:
```
uvicorn backend.api.server:app --reload --port 8000
open frontend/index.html
```
or visit `http://localhost:8000/` directly (the same FastAPI process now
serves the frontend too).

## Cost expectation

Railway's $5 starter plan is enough — the service idles cheaply between
requests, the cron is one ~10-min job per day, and the volume is tiny.
