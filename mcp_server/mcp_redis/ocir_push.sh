#    #!/usr/bin/env bash


# Replace values accordingly
# REG_URL="iad.ocir.io"
# NS="orasenatdpltintegration03"
# USER_PATH="oracleidentitycloudservice/anup.ojah@oracle.com"                         # e.g. aojah  (local)  or oracleidentitycloudservice/aojah@oracle.com (federated)
# echo "ssws" | podman login "$REG_URL" -u "${NS}/${USER_PATH}" --password-stdin
#chmod +x ocir_pull_run.sh
#./ocir_pull_run.sh



#!/usr/bin/env bash
set -euo pipefail

# ocir_build_push.sh — Build & push an image to OCIR using Podman
# Required vars (env or prompts):
#   REGION        e.g., iad
#   NAMESPACE     Object Storage namespace (NOT tenancy name)
#   USERNAME      OCI username (local) or "<domain>/<user>" if federated
#   AUTH_TOKEN    OCI auth token (NOT console password)
#   REPO          Repository name (lowercase recommended), e.g., db-operator
# Optional:
#   TAG           Defaults to "latest"
#   DOCKERFILE    Defaults to "Dockerfile"
#   CONTEXT       Defaults to "."
#   EXTRA_TAGS    If set to "1", also tag with date and git short SHA (if available)

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

# Collect inputs (env takes precedence; otherwise prompt if interactive)
prompt_if_missing REGION     "Enter OCIR region (e.g., iad, kix)"
prompt_if_missing NAMESPACE  "Enter Object Storage namespace (e.g., orasenatdpltintegration03)"
prompt_if_missing USERNAME   "Enter OCI username (for federated: oracleidentitycloudservice/email)"
secret_prompt_if_missing AUTH_TOKEN "Enter OCI Auth Token (input hidden)"
prompt_if_missing REPO       "Enter repo name (e.g., db-operator-mcp-redis)"

TAG="${TAG:-latest}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"
CONTEXT="${CONTEXT:-.}"
EXTRA_TAGS="${EXTRA_TAGS:-0}"

# Build registry URL and normalized repo
REG_URL="${REGION}.ocir.io"
REPO_LC="$(printf '%s' "${REPO}" | tr '[:upper:]' '[:lower:]')"
IMAGE="${REG_URL}/${NAMESPACE}/${REPO_LC}:${TAG}"

echo "==> Using registry    : ${REG_URL}"
echo "==> Using repository  : ${NAMESPACE}/${REPO_LC}"
echo "==> Using tag         : ${TAG}"
echo "==> Dockerfile        : ${DOCKERFILE}"
echo "==> Context           : ${CONTEXT}"

# On macOS with Podman Machine, ensure the VM is running
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

# Avoid proxy interference with OCIR (optional but helpful)
unset http_proxy HTTPS_PROXY https_proxy HTTP_PROXY || true

# Optional endpoint sanity check (401 is expected if not authed)
if command -v curl >/dev/null 2>&1; then
  echo "==> Probing registry endpoint..."
  curl -sSf -I "https://${REG_URL}/v2/" | head -n 1 || true
fi

# Login (don’t echo token)
echo "==> Logging into ${REG_URL} as ${NAMESPACE}/${USERNAME}"
# shellcheck disable=SC2310
if ! printf '%s' "${AUTH_TOKEN}" | podman login "${REG_URL}" -u "${NAMESPACE}/${USERNAME}" --password-stdin; then
  die "Login failed. Check namespace/username format and auth token."
fi

# Build
echo "==> Building image ${IMAGE}"
podman build -t "${IMAGE}" -f "${DOCKERFILE}" "${CONTEXT}"

# Extra tags (date + git sha) if requested
if [ "${EXTRA_TAGS}" = "1" ]; then
  DATE_TAG="$(date +%Y-%m-%d)"
  podman tag "${IMAGE}" "${REG_URL}/${NAMESPACE}/${REPO_LC}:${DATE_TAG}" || true
  if command -v git >/dev/null 2>&1 && git rev-parse --short HEAD >/dev/null 2>&1; then
    SHA_TAG="sha-$(git rev-parse --short HEAD)"
    podman tag "${IMAGE}" "${REG_URL}/${NAMESPACE}/${REPO_LC}:${SHA_TAG}" || true
    EXTRA_LIST="${DATE_TAG}, ${SHA_TAG}"
  else
    EXTRA_LIST="${DATE_TAG}"
  fi
  echo "==> Extra tags applied: ${EXTRA_LIST}"
fi

# Push primary tag
echo "==> Pushing ${IMAGE}"
podman push "${IMAGE}"

# Push extra tags if any
if [ "${EXTRA_TAGS}" = "1" ]; then
  [ -n "${DATE_TAG:-}" ] && podman push "${REG_URL}/${NAMESPACE}/${REPO_LC}:${DATE_TAG}" || true
  [ -n "${SHA_TAG:-}" ]  && podman push "${REG_URL}/${NAMESPACE}/${REPO_LC}:${SHA_TAG}"  || true
fi

echo "✅ Done. Image available at: ${IMAGE}"
# Clean up sensitive var
unset AUTH_TOKEN || true


