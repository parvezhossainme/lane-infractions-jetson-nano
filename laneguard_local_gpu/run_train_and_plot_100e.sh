#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_train_100e.sh"
PLOT_SCRIPT="${SCRIPT_DIR}/plot_curves.py"
OUT_DIR="${ROOT_DIR}/outputs/laneguard_local_gpu_100e"
LOG_FILE="${OUT_DIR}/train.log"
EPOCH_LOG_FILE="${OUT_DIR}/log.txt"
PLOT_FILE="${OUT_DIR}/train_test_graph.png"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python virtualenv not found at ${VENV_PYTHON}"
  exit 1
fi

bash "${TRAIN_SCRIPT}"

"${VENV_PYTHON}" "${PLOT_SCRIPT}" \
  --log-file "${EPOCH_LOG_FILE}" \
  --output-image "${PLOT_FILE}" \
  --title "Laneguard RT-DETRv2 R18 Training/Test Curves"

echo "Graph saved to: ${PLOT_FILE}"
