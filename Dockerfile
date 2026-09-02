FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY Backend ./Backend
EXPOSE 8000
CMD ["sh", "-c", "uvicorn Backend.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
