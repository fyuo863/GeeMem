#!/usr/bin/env bash
# Run as agent on geeai-aiagent. Fetch only from the configured public repository.
set -euo pipefail
base=/home/agent/csig-aml
repo="$base/repository"
url=https://github.com/fyuo863/GeeMem.git
branch=codex/agentmemories-evaluation
proxy=http://127.0.0.1:18093
expected="${1:?Usage: bash deploy/update-geeai.sh FULL_COMMIT_SHA}"
[[ "$expected" =~ ^[0-9a-f]{40}$ ]] || { echo 'A full commit SHA is required' >&2; exit 2; }
# Only one deployment may modify current at a time.
exec 9>"$base/deployment.lock"
flock -n 9 || { echo 'Another deployment is active' >&2; exit 1; }
if [[ ! -d "$repo/.git" ]]; then
    git -c http.proxy="$proxy" clone --branch "$branch" --single-branch "$url" "$repo"
fi
[[ "$(git -C "$repo" remote get-url origin)" == "$url" ]] || { echo 'Unexpected repository origin' >&2; exit 1; }
git -C "$repo" -c http.proxy="$proxy" fetch origin "$branch"
commit=$(git -C "$repo" rev-parse 'FETCH_HEAD^{commit}')
[[ "$commit" == "$expected" ]] || { echo 'Remote branch does not match the reviewed commit' >&2; exit 1; }
release="$base/releases/$commit"
if [[ ! -f "$release/.deployed-commit" ]]; then
    mkdir -p "$release"
    git -C "$repo" archive --format=tar "$commit" | tar -xf - -C "$release"
    ln -s "$base/shared/.env" "$release/.env"
    printf '%s\n' "$commit" > "$release/.deployed-commit"
fi
[[ "$(cat "$release/.deployed-commit")" == "$commit" ]] || exit 1
# This updater intentionally leaves runtime, credentials and database untouched.
# If dependencies change, prepare and validate a matching runtime before switching.
"$base/.venv/bin/python" - "$release/deploy/geeai-requirements.lock" <<'PY'
import importlib.metadata,sys
from pathlib import Path
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line or line.startswith('#'): continue
    name,version=line.split('==',1)
    if importlib.metadata.version(name)!=version:
        raise SystemExit('Runtime dependency mismatch: '+name)
PY
(cd "$release" && "$base/.venv/bin/python" -c 'import memory.aml_api')
previous=$(readlink -f "$base/current")
rollback() {
    trap - ERR
    ln -s "$previous" "$base/current.rollback"
    mv -Tf "$base/current.rollback" "$base/current"
    systemctl --user restart csig-aml.service
    echo 'Deployment failed; previous code release restored' >&2
}
trap rollback ERR
ln -s "$release" "$base/current.next"
mv -Tf "$base/current.next" "$base/current"
systemctl --user restart csig-aml.service
healthy=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if curl --noproxy '*' -fsS --max-time 3 http://127.0.0.1:18092/health >/dev/null; then healthy=1; break; fi
    sleep 1
done
[[ "$healthy" == 1 ]]
systemctl --user is-active --quiet csig-aml.service
trap - ERR
printf 'DEPLOYED_COMMIT=%s\n' "$commit"
