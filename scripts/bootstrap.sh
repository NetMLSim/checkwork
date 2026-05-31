#!/usr/bin/env bash
#
# CheckWork bootstrap.
#
# Builds the `chakra_env` Python virtual environment, installs Chakra
# (from upstream GitHub) and a compatible protobuf runtime, and then runs
# a one-line sanity check.
#
# Run from the repo root:
#     ./scripts/bootstrap.sh
#
# For the full, manual install (including building the ASTRA-sim Docker
# image), see INSTALLATION.md.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/chakra_env"

# Pick a Python interpreter (prefer 3.14, fall back to system python3).
PYTHON="${PYTHON:-$(command -v python3.14 || command -v python3)}"
if [[ -z "${PYTHON}" ]]; then
  echo "error: no python3 interpreter found on PATH" >&2
  exit 1
fi
echo "==> using python: ${PYTHON} ($(${PYTHON} --version))"

if [[ -d "${VENV_DIR}" ]]; then
  echo "==> chakra_env already exists. Re-using."
  echo "    (if you migrated machines and imports fail, delete the dir and re-run this script)"
else
  echo "==> creating venv at ${VENV_DIR}"
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

# Activate by sourcing the venv-relative python directly to avoid messing
# with the caller's shell state.
VENV_PY="${VENV_DIR}/bin/python"

echo "==> upgrading pip / wheel"
"${VENV_PY}" -m pip install --upgrade pip wheel >/dev/null

echo "==> installing core dependencies"
"${VENV_PY}" -m pip install --upgrade pyyaml numpy pandas matplotlib >/dev/null

echo "==> installing chakra from upstream GitHub"
"${VENV_PY}" -m pip install --upgrade "git+https://github.com/astra-sim/chakra.git" >/dev/null

echo "==> pinning protobuf to >=6.33,<7 (overrides chakra's protobuf==5.* constraint)"
"${VENV_PY}" -m pip install --upgrade "protobuf>=6.33,<7" >/dev/null

echo "==> sanity check"
"${VENV_PY}" - <<'PY'
import chakra  # noqa: F401
import google.protobuf as p
import yaml  # noqa: F401
import numpy, pandas, matplotlib  # noqa: F401
print(f"chakra ok | protobuf {p.__version__} | numpy {numpy.__version__} | matplotlib {matplotlib.__version__}")
PY

cat <<EOF

CheckWork Python environment is ready.

Next steps:
  1. Activate the venv in your shell:        source chakra_env/bin/activate
  2. Build the ASTRA-sim Docker image:       see INSTALLATION.md §4a
  3. Run the paper reproduction:             cd experiment/checkfreq_robustness && python3 run_robustness.py --quick --local
EOF
