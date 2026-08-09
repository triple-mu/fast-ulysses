#!/usr/bin/env bash
# Put this tree somewhere a cluster job can see, and leave behind what it needs to be attributable.
#
# .git is deliberately not copied (it is most of the size and none of the use), so the commit is
# written to a COMMIT file instead -- benchmark/collect.sh reads it when git is unavailable, and
# refuses to call a run attributable when neither is present.
#
#   tools/sync_to_cluster.sh <ssh-host> <remote-dir>
set -euo pipefail

HOST="${1:?usage: $0 <ssh-host> <remote-dir>}"
DEST="${2:?usage: $0 <ssh-host> <remote-dir>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

dirty="$(git -C "${REPO}" status --porcelain | wc -l)"
stamp="$(git -C "${REPO}" rev-parse --short HEAD)"
if ((dirty)); then
    stamp="${stamp}-dirty(${dirty})"
    echo "warning: ${dirty} uncommitted file(s); the stamp says so, but prefer committing first" >&2
fi
echo "${stamp}" > "${REPO}/COMMIT"

rsync -az --delete -e "ssh -o RemoteCommand=none" \
    --exclude '.git' --exclude 'build' --exclude '__pycache__' --exclude '*.so' \
    --exclude '*.egg-info' --exclude '3rdparty' --exclude 'cmake-build-debug' \
    --exclude '.ruff_cache' --exclude '.idea' --exclude 'benchmark-results' \
    "${REPO}/" "${HOST}:${DEST}/"

rm -f "${REPO}/COMMIT"
echo "synced ${stamp} -> ${HOST}:${DEST}"
