#!/usr/bin/env bash
# scripts/setup_faiss_env.sh
# Purpose: Stand up a working FAISS (CPU) environment for the live FastAPI+FAISS stack
# Targets: macOS arm64 (Apple Silicon) primary, Linux x86_64 secondary
# Defaults: NO_DOCKER; uses conda-forge first, then source build; Rosetta fallback optional
# Usage:
#   ./scripts/setup_faiss_env.sh [--strategy conda|source|rosetta|system] [--env lnsp-faiss] [--py 3.11] [--force]
# Notes:
#   - Creates/activates a Python env and installs faiss-cpu
#   - Verifies by importing faiss and building a tiny IVF index
#   - Leaves a stamp file at .faiss_ready to signal success

set -euo pipefail

STRATEGY="${1:-}"
ENV_NAME="lnsp-faiss"
PY_VERSION="3.11"
FORCE_RECREATE="false"

# ANSI colors
BOLD="$(tput bold || true)"
DIM="$(tput dim || true)"
RED="$(tput setaf 1 || true)"
GRN="$(tput setaf 2 || true)"
YLW="$(tput setaf 3 || true)"
CYN="$(tput setaf 6 || true)"
RST="$(tput sgr0 || true)"

log() { echo "${CYN}[faiss-setup]${RST} $*"; }
ok()  { echo "${GRN}[ok]${RST} $*"; }
warn(){ echo "${YLW}[warn]${RST} $*"; }
err() { echo "${RED}[err]${RST} $*" >&2; }

usage() {
  cat <<EOF
${BOLD}FAISS Environment Setup${RST}
${DIM}NO_DOCKER by default; picks best strategy per platform${RST}

${BOLD}Options:${RST}
  --strategy {conda|source|rosetta|system}
      conda    : Use conda-forge faiss-cpu (recommended on macOS arm64, py${PY_VERSION})
      source   : Build FAISS from source (OpenBLAS/Accelerate)
      rosetta  : Create x86_64 Python env on macOS via Rosetta and pip-install faiss-cpu
      system   : Try pip-only in current env (best-effort)
  --env NAME   : Environment name (default: ${ENV_NAME})
  --py VER     : Python version (default: ${PY_VERSION})
  --force      : Recreate env if exists

Examples:
  ./scripts/setup_faiss_env.sh
  ./scripts/setup_faiss_env.sh --strategy conda --py 3.11
  ./scripts/setup_faiss_env.sh --strategy rosetta --py 3.9
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy) STRATEGY="$2"; shift 2;;
    --env) ENV_NAME="$2"; shift 2;;
    --py) PY_VERSION="$2"; shift 2;;
    --force) FORCE_RECREATE="true"; shift;;
    -h|--help) usage; exit 0;;
    *) shift;;
  esac
done

OS="$(uname -s || true)"
ARCH="$(uname -m || true)"

detect_platform() {
  log "Platform: OS=${OS}, ARCH=${ARCH}, Python target=${PY_VERSION}"
  if [[ "${OS}" == "Darwin" && "${ARCH}" == "arm64" ]]; then
    echo "macos_arm64"
  elif [[ "${OS}" == "Linux" ]]; then
    echo "linux"
  else
    echo "unknown"
  fi
}

ensure_conda() {
  if command -v conda >/dev/null 2>&1; then
    ok "conda found: $(conda --version)"
  else
    warn "conda not found. Installing Miniforge (recommended for Apple Silicon)…"
    # Miniforge (conda-forge) for clean arm64 support
    TMPD="$(mktemp -d)"
    curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh \
      -o "${TMPD}/miniforge.sh"
    bash "${TMPD}/miniforge.sh" -b -p "$HOME/miniforge3"
    rm -rf "${TMPD}"
    # shellcheck disable=SC1091
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    ok "Miniforge installed."
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
}

create_conda_env() {
  local name="$1" py="$2"
  if conda env list | awk '{print $1}' | grep -qx "${name}"; then
    if [[ "${FORCE_RECREATE}" == "true" ]]; then
      warn "Env ${name} exists; --force specified, removing…"
      conda env remove -n "${name}" -y || true
    else
      ok "Reusing existing conda env: ${name}"
    fi
  fi
  if ! conda env list | awk '{print $1}' | grep -qx "${name}"; then
    log "Creating conda env ${name} (python=${py})…"
    conda create -y -n "${name}" "python=${py}"
  fi
  conda activate "${name}"
  ok "Activated conda env: ${name}"
}

install_faiss_conda() {
  log "Installing faiss-cpu from conda-forge…"
  conda install -y -c conda-forge faiss-cpu numpy scipy
  ok "faiss-cpu installed via conda-forge."
}

install_deps_macos_source() {
  log "Ensuring macOS build deps (Homebrew)…"
  if ! command -v brew >/dev/null 2>&1; then
    err "Homebrew not found. Install Homebrew or use --strategy conda"; exit 1
  fi
  brew install cmake ninja pkg-config openblas || true
  ok "macOS build deps ready."
}

install_deps_linux_source() {
  log "Ensuring Linux build deps (APT)…"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y build-essential cmake ninja-build pkg-config libopenblas-dev python3-dev python3-venv
    ok "Linux build deps ready."
  else
    warn "Non-APT distro; ensure cmake, ninja, OpenBLAS, python dev headers are installed."
  fi
}

create_venv_if_none() {
  if [[ -d ".venv" && "${FORCE_RECREATE}" == "true" ]]; then
    warn "Removing existing .venv due to --force…"
    rm -rf .venv
  fi
  if [[ ! -d ".venv" ]]; then
    log "Creating virtualenv .venv (python${PY_VERSION})…"
    python${PY_VERSION} -m venv .venv || python -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip wheel setuptools
  ok "Activated venv: .venv"
}

install_faiss_pip_best_effort() {
  log "Trying pip install faiss-cpu (best-effort)…"
  pip install "faiss-cpu" numpy || {
    warn "pip faiss-cpu failed on this platform. Consider --strategy conda or --strategy source"
    return 1
  }
  ok "faiss-cpu installed via pip."
}

build_faiss_from_source() {
  local faiss_dir=".faiss_src"
  log "Cloning FAISS source…"
  rm -rf "${faiss_dir}"
  git clone --depth=1 https://github.com/facebookresearch/faiss.git "${faiss_dir}"
  pushd "${faiss_dir}" >/dev/null

  # Build C++ core
  log "Configuring C++ core…"
  mkdir -p build
  pushd build >/dev/null
  # macOS: use Accelerate by default; otherwise use OpenBLAS
  local cmake_flags="-DFAISS_ENABLE_PYTHON=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=generic"
  if [[ "${OS}" == "Darwin" ]]; then
    cmake -G Ninja ${cmake_flags} -DFAISS_BLAS=Accelerate ..
  else
    cmake -G Ninja ${cmake_flags} -DFAISS_BLAS=OpenBLAS ..
  fi
  ninja
  sudo ninja install || true
  sudo ldconfig || true 2>/dev/null || true
  popd >/dev/null

  # Build Python bindings
  log "Building Python bindings…"
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
  pip install numpy cython
  pushd faiss/python >/dev/null
  python setup.py build
  python setup.py install
  popd >/dev/null

  popd >/dev/null
  ok "FAISS built from source and installed into current env."
}

setup_rosetta_env() {
  if [[ "${OS}" != "Darwin" ]]; then
    err "Rosetta strategy is macOS-only"; exit 1
  fi
  log "Setting up Rosetta x86_64 venv (allows pip faiss-cpu wheels)…"
  if ! /usr/sbin/sysctl -n sysctl.proc_translated >/dev/null 2>&1; then
    warn "Rosetta translation status unknown; ensure Rosetta 2 is installed: softwareupdate --install-rosetta --agree-to-license"
  fi
  # Create x86_64 venv
  /usr/bin/arch -x86_64 /usr/bin/python${PY_VERSION} -m venv .venv_x86 || \
  /usr/bin/arch -x86_64 /usr/bin/python3 -m venv .venv_x86
  # shellcheck disable=SC1091
  source .venv_x86/bin/activate
  /usr/bin/arch -x86_64 python -m pip install --upgrade pip wheel setuptools
  /usr/bin/arch -x86_64 python -m pip install numpy "faiss-cpu"
  ok "Installed faiss-cpu in Rosetta venv (.venv_x86)."
}

verify_faiss() {
  log "Verifying FAISS import and tiny IVF index…"
  python - <<'PYCODE'
import sys, numpy as np
try:
    import faiss
except Exception as e:
    print("FAISS import failed:", e, file=sys.stderr); sys.exit(2)
d = 64
nb = 2000
nq = 5
np.random.seed(0)
xb = np.random.randn(nb, d).astype('float32')
xq = np.random.randn(nq, d).astype('float32')
# Normalize for cosine-equivalent IP search (optional)
faiss.normalize_L2(xb)
faiss.normalize_L2(xq)
# IVF Flat
quantizer = faiss.IndexFlatIP(d)
index = faiss.IndexIVFFlat(quantizer, d, 64, faiss.METRIC_INNER_PRODUCT)
index.train(xb)
index.add(xb)
D, I = index.search(xq, 3)
print("FAISS ok. Top-3 IDs for 5 queries:\n", I)
PYCODE
  ok "FAISS verification succeeded."
  touch .faiss_ready
}

platform="$(detect_platform)"

# Default strategy selection if none provided
if [[ -z "${STRATEGY}" || "${STRATEGY}" == "--strategy" ]]; then
  case "${platform}" in
    macos_arm64) STRATEGY="conda";;
    linux) STRATEGY="conda";;
    *) STRATEGY="system";;
  esac
fi

log "Selected strategy: ${BOLD}${STRATEGY}${RST}"
echo "::strategy=${STRATEGY}"

case "${STRATEGY}" in
  conda)
    ensure_conda
    create_conda_env "${ENV_NAME}" "${PY_VERSION}"
    install_faiss_conda
    verify_faiss
    ;;
  source)
    if [[ "${OS}" == "Darwin" ]]; then
      install_deps_macos_source
    else
      install_deps_linux_source
    fi
    create_venv_if_none
    build_faiss_from_source
    verify_faiss
    ;;
  rosetta)
    setup_rosetta_env
    verify_faiss
    ;;
  system)
    # Best-effort in current environment
    log "Attempting pip-only install of faiss-cpu in current env…"
    pip install --upgrade pip wheel setuptools numpy || true
    if ! install_faiss_pip_best_effort; then
      err "pip faiss-cpu failed. Try: --strategy conda  OR  --strategy source  OR  --strategy rosetta (macOS only)."
      exit 2
    fi
    verify_faiss
    ;;
  *)
    err "Unknown strategy: ${STRATEGY}"; usage; exit 1;;
esac

ok "FAISS environment is ready. Stamp file: .faiss_ready"
echo
echo "${BOLD}Next steps:${RST}
  1) Activate your environment:
     ${DIM}# if conda strategy${RST}
     conda activate ${ENV_NAME}
     ${DIM}# if source/system strategy with venv${RST}
     source .venv/bin/activate
     ${DIM}# if rosetta strategy${RST}
     source .venv_x86/bin/activate

  2) Rebuild and load your FAISS index:
     make build-faiss || PYTHONPATH=src python src/faiss_index.py

  3) Start API and re-run consultant eval:
     PYTHONPATH=src uvicorn src.api.retrieve:app --port 8092
     make consultant-eval
"
