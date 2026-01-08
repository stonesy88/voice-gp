FROM python:3.13

WORKDIR /app

RUN pip install --no-cache-dir \
    torch \
    sentence-transformers \
    fastapi \
    uvicorn

RUN pip install --no-cache-dir \
    #neo4j==6.0.3 \
    neo4j-rust-ext==6.0.3.0

COPY server.py .
COPY ingest_snomed.py . 

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]