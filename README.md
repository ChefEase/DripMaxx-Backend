## DripMaxx API (Phase 3.3 - AI scoring)

FastAPI service with Postgres/Supabase schema plus Replicate-powered CLIP embeddings for Drip scoring.

### Run locally
```bash
cd api
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Environment
- `DATABASE_URL` (Postgres/Supabase) is required. Example: `postgresql+asyncpg://postgres:postgres@localhost:5432/postgres`.
- `REPLICATE_API_TOKEN` is required for live scoring.
- `REPLICATE_MODEL` defaults to `krthr/clip-embeddings:1c0371070cb827ec3c7f2f28adcdde54b50dcd239aa6faea0bc98b174ef03fb4` but can be overridden.
- On startup we run `Base.metadata.create_all()` to sync the schema for local dev; against Supabase this is a no-op if tables already exist.

### Endpoints
- `GET /health` - health check.
- `POST /v1/outfits/score` - multipart form with:
  - `image` file (jpg/png)
  - `user_context` field containing JSON. Accepts either:
    - `{"style_preferences":["streetwear"],"style_inspirations":["ASAP Rocky"],"user_height":"180","user_body_type":"athletic","gender_style_preference":"menswear"}` **or**
    - `{"user_context":{...same fields...}}`
  Calls Replicate to fetch CLIP embeddings, computes weighted Drip Score + breakdown, then asks a Replicate Llama-4 instruct model for **15 suggestion cards**. Each card includes `title`, `type`, `description`, and attempts an image using `google/nano-banana` conditioned on the uploaded outfit. Falls back to heuristic suggestions if AI is unavailable.

### Notes
- Logging via Loguru to stdout; AI scoring logs include brightness/contrast and breakdown for debugging (no raw embeddings).
- CORS is open for development.
- See `.env.example` for required secrets.
