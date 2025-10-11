#!/usr/bin/env bash
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }

prompt_if_missing() {
  local var="$1" prompt="$2"
  if [ -z "${!var:-}" ]; then
    if [ -t 0 ]; then
      read -rp "$prompt: " "$var" || die "Failed reading $var"
      export "$var"
    else
      die "$var is required"
    fi
  fi
}

secret_prompt_if_missing() {
  local var="$1" prompt="$2"
  if [ -z "${!var:-}" ]; then
    if [ -t 0 ]; then
      printf "%s: " "$prompt" >&2
      stty -echo
      IFS= read -r __secret || { stty echo; echo; die "Failed reading $var"; }
      stty echo
      echo
      printf -v "$var" '%s' "$__secret"
      export "$var"
    else
      die "$var is required"
    fi
  fi
}

# ======= Inputs =======
# Actions: build-push-run (default), push-run, pull-run
ACTION="${ACTION:-${1:-build-push-run}}"

prompt_if_missing REGION     "Enter OCIR region (e.g., iad, kix)"
prompt_if_missing NAMESPACE  "Enter Object Storage namespace (e.g., orasenatdpltintegration03)"
prompt_if_missing USERNAME   "Enter OCI username (for federated: oracleidentitycloudservice/<email>)"
secret_prompt_if_missing AUTH_TOKEN "Enter OCI Auth Token (input hidden)"
prompt_if_missing REPO       "Enter repo name (e.g., db-operator-mcp-redis)"

TAG="${TAG:-latest}"
PORT_MAP="${PORT_MAP:-8001:8001}"
PLATFORM="${PLATFORM:-}"                      # e.g. linux/amd64 on Apple Silicon
REDIS_URL="${REDIS_URL:-}"                    # optional for your app
LOCAL_IMAGE="${LOCAL_IMAGE:-${REPO}:local}"   # local image tag to build/push
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CONTEXT="${CONTEXT:-.}"

# ======= Derived =======
REG_URL="${REGION}.ocir.io"
REPO_LC="$(printf '%s' "${REPO}" | tr '[:upper:]' '[:lower:]')"
REMOTE_IMAGE="${REG_URL}/${NAMESPACE}/${REPO_LC}:${TAG}"
CONTAINER_NAME="$(basename "${REPO_LC}")"

echo "==> Podman only"
# ========= Podman sanity =========
command -v podman >/dev/null 2>&1 || die "podman not found in PATH"

# On macOS/Windows, only try to start the VM if it's not running
if command -v podman-machine >/dev/null 2>&1 || podman machine --help >/dev/null 2>&1; then
  # Try a cheap “is it usable?” test first
  if ! podman info >/dev/null 2>&1; then
    # If info fails, check VM state; only start if not running
    state="$(podman machine list --format '{{.Name}} {{.Running}}' 2>/dev/null | awk '/default/ {print $2}')"
    if [ "${state}" != "true" ]; then
      echo "==> Starting Podman machine..."
      podman machine start >/dev/null 2>&1 || true
    fi
  fi
fi


echo "==> Registry   : ${REG_URL}"
echo "==> Repository : ${NAMESPACE}/${REPO_LC}"
echo "==> Tag        : ${TAG}"
echo "==> Remote     : ${REMOTE_IMAGE}"
echo "==> Local      : ${LOCAL_IMAGE}"
echo "==> Action     : ${ACTION}"

# Avoid proxy issues with registry
unset http_proxy HTTPS_PROXY https_proxy HTTP_PROXY || true

# Optional probe
if command -v curl >/dev/null 2>&1; then
  echo "==> Probing ${REG_URL} ..."
  curl -sSf -I "https://${REG_URL}/v2/" | head -n 1 || true
fi

# ======= Login =======
echo "==> Logging into ${REG_URL} as ${NAMESPACE}/${USERNAME}"
if ! printf '%s' "${AUTH_TOKEN}" | podman login "${REG_URL}" -u "${NAMESPACE}/${USERNAME}" --password-stdin; then
  die "Login failed. Check namespace/username and auth token."
fi

# ======= Helpers =======
pm_build() {
  echo "==> Building ${LOCAL_IMAGE} (Dockerfile=${DOCKERFILE}, context=${CONTEXT})"
  podman build -t "${LOCAL_IMAGE}" -f "${DOCKERFILE}" "${CONTEXT}"
}

pm_push() {
  echo "==> Tagging ${LOCAL_IMAGE} -> ${REMOTE_IMAGE}"
  podman image exists "${LOCAL_IMAGE}" || die "Local image '${LOCAL_IMAGE}' not found. Build first or set LOCAL_IMAGE."
  podman tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
  echo "==> Pushing ${REMOTE_IMAGE}"
  podman push "${REMOTE_IMAGE}"
}

pm_pull() {
  echo "==> Pulling ${REMOTE_IMAGE}"
  if [ -n "${PLATFORM}" ]; then
    podman pull --platform "${PLATFORM}" "${REMOTE_IMAGE}"
  else
    podman pull "${REMOTE_IMAGE}"
  fi
}

pm_run() {
  echo "==> Running ${CONTAINER_NAME}"
  ARGS=(--rm -it --name "${CONTAINER_NAME}" -p "${PORT_MAP}"
        -e MCP_SSE_HOST=0.0.0.0 -e MCP_SSE_PORT=8001)
  [ -n "${REDIS_URL}" ] && ARGS+=(-e "REDIS_URL=${REDIS_URL}")
  [ -n "${PLATFORM}" ] && ARGS+=(--platform "${PLATFORM}")
  podman run "${ARGS[@]}" "${REMOTE_IMAGE}"
}

pm_inspect() {
  echo "==> Inspect:"
  podman inspect --format '{{.Config.ExposedPorts}} {{.Config.Entrypoint}} {{.Config.Cmd}}' "${REMOTE_IMAGE}" || true
}

# ======= Actions =======
case "${ACTION}" in
  build-push-run)
    pm_build
    pm_push
    pm_pull
    pm_inspect
    pm_run
    ;;
  push-run)
    pm_push
    pm_pull
    pm_inspect
    pm_run
    ;;
  pull-run)
    pm_pull
    pm_inspect
    pm_run
    ;;
  *)
    die "Unknown ACTION '${ACTION}'. Use: build-push-run | push-run | pull-run"
    ;;
esac
