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
   ADMIN_TOKEN=<a long random string you generate>
   ```
   - Get a Leaguepedia bot password at https://lol.fandom.com/wiki/Special:BotPasswords
     (grants needed: "Basic rights" + "High-volume editing").
   - `ADMIN_TOKEN` lets you trigger ingestion remotely without SSH (see step 5).
     Generate one with `openssl rand -hex 32` or any password manager.

5. **First deploy** boots with an empty DB (`/health` returns 0 matches).
   Trigger ingestion + retrain from your laptop:
   ```bash
   APP=https://<your-app>.up.railway.app
   TOKEN=<the ADMIN_TOKEN you set>

   # kick off (returns immediately)
   curl -X POST -H "X-Admin-Token: $TOKEN" $APP/admin/refresh

   # follow progress live (~10 minutes)
   curl -N -H "X-Admin-Token: $TOKEN" $APP/admin/refresh/stream

   # or poll status with the last 50 log lines
   curl -H "X-Admin-Token: $TOKEN" $APP/admin/refresh/status
   ```
   After it finishes, `/health` shows real numbers and the frontend works.

## Visiting the app

The Railway URL serves both:
- `https://<app>.railway.app/` — frontend
- `https://<app>.railway.app/health`, `/predict`, etc — API

The frontend autodetects the host, so no config needed.

## Keeping data fresh

Two options:

**Cron (server-side)** — Railway cron job (Settings → Cron):
- Schedule: `0 12 * * *` (daily at noon UTC) — fine for tracking new patches
- Command: `python -m backend.refresh`

**On-demand (client-side)** — re-run `curl -X POST -H "X-Admin-Token: $TOKEN"
$APP/admin/refresh` whenever you want fresh data. Idempotent.

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
