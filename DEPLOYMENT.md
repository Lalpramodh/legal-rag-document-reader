# Legal RAG deployment

## Backend on Render
Deploy the repository as a Docker Web Service.

Environment variables:
- `MONGO_URI` = MongoDB Atlas SRV URI
- `MONGO_DB` = `legal_assistant`
- `ALLOWED_ORIGINS` = deployed Vercel URL (comma-separated if multiple)
- `GROQ_API_KEY` = Groq API key
- `GROQ_MODEL` = `openai/gpt-oss-20b` (optional; the backend verifies availability)

Health endpoint: `/health`

## Frontend on Vercel
- Root directory: `Frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com`

Never commit `.env` or secrets to GitHub.

The Docker service uses the repository root as its context and starts with
`uvicorn Backend.app:app --host 0.0.0.0 --port $PORT`. Render's Docker service
automatically provides `PORT`; no manual port setting is required.
