#!/usr/bin/env bash
# Start the RAG stack on the EC2 server. Run by rag-stack.service at every boot and by CD.
# Secrets are read from SSM and written only to /run (memory); they are never echoed or put on disk.
set -euo pipefail
umask 077
. /etc/rag/rag.conf
export AWS_DEFAULT_REGION="$REGION"

get() { aws ssm get-parameter --name "$1" --with-decryption --query Parameter.Value --output text; }

TAG="$(get /rag/desired_image_tag)"
if [ "$TAG" = "none" ]; then
  echo "desired_image_tag is 'none': no image deployed yet, stack not started."
  exit 0
fi

# Assign first so a failed lookup stops the script (set -e ignores failures inside $(...) arguments).
API_KEY="$(get /rag/qwen_api_key)"
API_BASE="$(get /rag/qwen_base_url)"
NGROK_TOKEN="$(get /rag/ngrok_authtoken)"
NGROK_HOST="$(get /rag/ngrok_domain)"
ORIGIN="$(get /rag/frontend_origin)"
PROVIDER="$(get /rag/provider_enabled)"

install -d -m 0700 /run/rag /run/rag/docker
export DOCKER_CONFIG=/run/rag/docker
aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_REGISTRY" > /dev/null

{
  printf "RAG_IMAGE='%s'\n" "$ECR_REGISTRY/$ECR_REPOSITORY:$TAG"
  printf "LIVE_QWEN_API_KEY='%s'\n" "$API_KEY"
  printf "LIVE_QWEN_API_BASE='%s'\n" "$API_BASE"
  printf "NGROK_AUTHTOKEN='%s'\n" "$NGROK_TOKEN"
  printf "NGROK_DOMAIN='%s'\n" "$NGROK_HOST"
  printf "LIVE_FRONTEND_ORIGIN='%s'\n" "$ORIGIN"
  printf "LIVE_PROVIDER_ENABLED='%s'\n" "$PROVIDER"
} > /run/rag/env

docker compose -f /opt/rag/compose.aws.yaml --env-file /run/rag/env up -d --remove-orphans

# Auto-stop safety net: a forgotten server shuts itself down (EC2 then stops, it is not deleted).
shutdown -c 2> /dev/null || true
shutdown -h "+${AUTOSTOP_MINUTES}" "RAG demo auto-stop" || true
echo "Stack started with image tag $TAG; auto-stop in ${AUTOSTOP_MINUTES} minutes."
