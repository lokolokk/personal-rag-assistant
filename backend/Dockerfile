FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY backend/requirements.lock .
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --only-binary=:all: --require-hashes -r requirements.lock
# Tokenizer vocabulary is public build data, not a model/provider dispatch.
ENV TIKTOKEN_CACHE_DIR=/opt/token-cache
RUN /opt/venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

FROM python:3.11.14-slim-bookworm@sha256:65a93d69fa75478d554f4ad27c85c1e69fa184956261b4301ebaf6dbb0a3543d AS runtime
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    BASE_DIR=/app STORAGE_DIR=/app/storage UPLOAD_DIR=/app/storage/uploads \
    CHROMA_PERSIST_DIR=/app/storage/chroma TIKTOKEN_CACHE_DIR=/opt/token-cache \
    ANONYMIZED_TELEMETRY=False HF_HUB_OFFLINE=1 RAG_COST_DEBUG=0
RUN groupadd --gid 10001 rag && useradd --uid 10001 --gid rag --no-create-home rag \
    && mkdir -p /app/storage/uploads /app/storage/chroma \
    && chown -R 10001:10001 /app/storage
COPY --from=dependencies /opt/venv /opt/venv
COPY --from=dependencies /opt/token-cache /opt/token-cache
WORKDIR /app/backend
COPY backend/app ./app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=4s --start-period=60s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).close()"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
