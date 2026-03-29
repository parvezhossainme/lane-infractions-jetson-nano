#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found at ${PYTHON_BIN}"
  echo "Create it first: python3 -m venv ${ROOT_DIR}/.venv"
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" UpdatedWay/detectors/OnlyLaneDetect.py
