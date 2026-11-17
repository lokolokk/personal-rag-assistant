#!/usr/bin/env bash
# Stop the RAG stack cleanly (flushes the database and vector index) before the server stops.
set -euo pipefail
# `down` needs these only to satisfy variable checks in the compose file.
RAG_IMAGE=unused NGROK_DOMAIN=unused NGROK_AUTHTOKEN=unused \
  docker compose -f /opt/rag/compose.aws.yaml down --timeout 60
