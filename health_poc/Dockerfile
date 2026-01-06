FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    neo4j \
    sentence-transformers \
    torch

COPY server.py .

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]