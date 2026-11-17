#!/usr/bin/env bash
# One-time server setup, run by cloud-init on the first boot (see infra/user_data.yaml.tftpl).
set -euo pipefail
. /etc/rag/rag.conf

# 2 GiB swap so a large PDF cannot trigger an out-of-memory kill on a 2 GiB machine.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile
  mkswap /swapfile > /dev/null
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
swapon -a

dnf install -y docker

# Docker Compose plugin: pinned version, checksum verified before install.
install -d /usr/local/lib/docker/cli-plugins
curl -fsSL -o /tmp/docker-compose \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"
echo "${COMPOSE_SHA256}  /tmp/docker-compose" | sha256sum -c -
install -m 0755 /tmp/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
rm -f /tmp/docker-compose

systemctl daemon-reload
systemctl enable --now docker
systemctl enable rag-stack.service
systemctl start rag-stack.service
