# Legal RAG deployment

## Backend on Render
Deploy the repository as a Docker Web Service.

Environment variables:
- `MONGO_URI` = MongoDB Atlas SRV URI
- `MONGO_DB` = `legal_assistant`
- `ALLOWED_ORIGINS` = deployed Vercel URL (comma-separated if multiple)

Health endpoint: `/health`

## Frontend on Vercel
- Root directory: `Frontend`
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_URL=https://YOUR-RENDER-SERVICE.onrender.com`

Never commit `.env` or secrets to GitHub.
