#!/usr/bin/env bash
# "Backup now": cold copy of the live data volume to the private S3 bucket (backups/).
# The backend is paused for the few seconds the copy takes, so the database and the vector index
# are consistent. It always resumes, even if something fails. Installed at /opt/rag/bin/rag-backup.sh.
set -euo pipefail
umask 077
. /etc/rag/rag.conf
export AWS_DEFAULT_REGION="$REGION"

CONTAINER=rag-backend-1
VOLUME=rag_live-storage
WORK=/run/rag-backup
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
NAME="live-storage-$STAMP.tgz"

BUCKET="$(aws ssm get-parameter --name /rag/data_bucket --query Parameter.Value --output text)"
docker inspect "$CONTAINER" > /dev/null # fails clearly if the stack was never started
IMG="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER")"

AVAIL_MB="$(df --output=avail -m /run | tail -n 1 | tr -d ' ')"
[ "$AVAIL_MB" -gt 200 ] || { echo "Not enough scratch space in /run (${AVAIL_MB} MB)."; exit 1; }

resume() {
  if [ "$(docker inspect --format '{{.State.Running}}' "$CONTAINER" 2> /dev/null)" != "true" ]; then
    docker start "$CONTAINER" > /dev/null
  fi
  for _ in $(seq 1 30); do
    [ "$(docker inspect --format '{{.State.Health.Status}}' "$CONTAINER" 2> /dev/null)" = "healthy" ] && return 0
    sleep 2
  done
  echo "WARNING: backend not healthy 60 s after resuming."
}
cleanup() {
  resume
  rm -rf "$WORK"
}
trap cleanup EXIT

rm -rf "$WORK"
install -d -m 0700 "$WORK"

docker stop --time 60 "$CONTAINER" > /dev/null
# Read-only, non-root, no network. The archive streams to the host's memory-backed /run.
docker run --rm --network none --user 10001:10001 --cap-drop ALL \
  -v "$VOLUME:/data:ro" --entrypoint tar "$IMG" --numeric-owner -cz -C /data . > "$WORK/$NAME"
resume

(cd "$WORK" && sha256sum "$NAME" > "$NAME.sha256")
aws s3 cp "$WORK/$NAME" "s3://$BUCKET/backups/$NAME" --only-show-errors
aws s3 cp "$WORK/$NAME.sha256" "s3://$BUCKET/backups/$NAME.sha256" --only-show-errors

LOCAL_SIZE="$(stat -c %s "$WORK/$NAME")"
REMOTE_SIZE="$(aws s3api head-object --bucket "$BUCKET" --key "backups/$NAME" --query ContentLength --output text)"
[ "$LOCAL_SIZE" = "$REMOTE_SIZE" ] || { echo "Upload size mismatch: $LOCAL_SIZE vs $REMOTE_SIZE"; exit 1; }

echo "Backup saved: s3://$BUCKET/backups/$NAME ($REMOTE_SIZE bytes)"
cut -d' ' -f1 "$WORK/$NAME.sha256" | sed 's/^/sha256: /'
