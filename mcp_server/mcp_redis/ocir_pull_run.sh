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
      IFS= read -r "$var" || die "Failed reading $var"
      stty echo
      echo
      export "$var"
    else
      die "$var is required"
    fi
  fi
}

# ========= Inputs =========
prompt_if_missing REGION     "Enter OCIR region (e.g., iad, kix)"
prompt_if_missing NAMESPACE  "Enter Object Storage namespace (e.g., orasenatdpltintegration03)"
prompt_if_missing USERNAME   "Enter OCI username (for federated: oracleidentitycloudservice/email)"
secret_prompt_if_missing AUTH_TOKEN "Enter OCI Auth Token (input hidden)"
prompt_if_missing REPO       "Enter repo name (e.g., db-operator-mcp-redis)"

TAG="${TAG:-latest}"
PLATFORM="${PLATFORM:-}"         # e.g. linux/amd64 on Apple Silicon
PORT_MAP="${PORT_MAP:-8003:8003}"
REDIS_URL="${REDIS_URL:-}"

# ========= Derived =========
REG_URL="${REGION}.ocir.io"
REPO_LC="$(printf '%s' "${REPO}" | tr '[:upper:]' '[:lower:]')"
IMAGE="${REG_URL}/${NAMESPACE}/${REPO_LC}:${TAG}"
CONTAINER_NAME="$(basename "${REPO_LC}")"

echo "==> Using registry   : ${REG_URL}"
echo "==> Using repository : ${NAMESPACE}/${REPO_LC}"
echo "==> Using tag        : ${TAG}"

# ========= Podman sanity =========
if command -v podman >/dev/null 2>&1; then
  if ! podman info >/dev/null 2>&1; then
    if command -v podman-machine >/dev/null 2>&1 || podman machine --help >/dev/null 2>&1; then
      echo "==> Starting Podman machine..."
      podman machine start >/dev/null
    fi
  fi
else
  die "podman not found in PATH"
fi

unset http_proxy HTTPS_PROXY https_proxy HTTP_PROXY || true

if command -v curl >/dev/null 2>&1; then
  echo "==> Probing registry endpoint..."
  curl -sSf -I "https://${REG_URL}/v2/" | head -n 1 || true
fi

# ========= Login =========
echo "==> Logging into ${REG_URL} as ${NAMESPACE}/${USERNAME}"
if ! printf '%s' "${AUTH_TOKEN}" | podman login "${REG_URL}" -u "${NAMESPACE}/${USERNAME}" --password-stdin; then
  die "Login failed. Check namespace/username format and auth token."
fi

# ========= Pull =========
echo "==> Pulling ${IMAGE}"
if [ -n "${PLATFORM}" ]; then
  podman pull --platform "${PLATFORM}" "${IMAGE}"
else
  podman pull "${IMAGE}"
fi

# ========= Inspect =========
echo "==> Exposed ports:"
podman inspect --format '{{.Config.ExposedPorts}}' "${IMAGE}" || true
echo "==> Entrypoint/Cmd:"
podman inspect --format '{{.Config.Entrypoint}} {{.Config.Cmd}}' "${IMAGE}" || true

# ========= Run =========
echo "==> Running container ${CONTAINER_NAME}"
RUN_ARGS=(--rm -it --name "${CONTAINER_NAME}")
[ -n "${PORT_MAP}" ] && RUN_ARGS+=(-p "${PORT_MAP}")
[ -n "${REDIS_URL}" ] && RUN_ARGS+=(-e "REDIS_URL=${REDIS_URL}")
[ -n "${PLATFORM}" ] && RUN_ARGS+=(--platform "${PLATFORM}")

podman run "${RUN_ARGS[@]}" "${IMAGE}"
