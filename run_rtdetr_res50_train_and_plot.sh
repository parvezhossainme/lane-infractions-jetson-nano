#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python virtualenv not found at ${PYTHON_BIN}"
  exit 1
fi

CONFIG_PATH="${ROOT_DIR}/rt-detr-backbones/res50/rtdetrv2_res50_6x_coco_eval.yml"
OUT_DIR="${ROOT_DIR}/outputs/backbone_benchmark/res50_coco8_train"
LOG_FILE="${OUT_DIR}/log.txt"
PLOT_FILE="${OUT_DIR}/rtdetr_res50_training_curves.png"

TRAIN_IMG="${ROOT_DIR}/datasets/coco8/images/train"
TRAIN_ANN="${ROOT_DIR}/datasets/coco8/annotations/instances_train2017.json"
VAL_IMG="${ROOT_DIR}/datasets/coco8/images/val"
VAL_ANN="${ROOT_DIR}/datasets/coco8/annotations/instances_val2017.json"

CHECKPOINT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"

mkdir -p "${OUT_DIR}"

echo "[1/2] Training RT-DETRv2 Res50 on coco8 (short run for plotting)..."
(
  cd "${ROOT_DIR}/RT-DETR/rtdetrv2_pytorch"
  "${PYTHON_BIN}" tools/train.py \
    -c "${CONFIG_PATH}" \
    -t "${CHECKPOINT_URL}" \
    -d cpu \
    --seed 0 \
    --output-dir "${OUT_DIR}" \
    -u \
      train_dataloader.dataset.img_folder="${TRAIN_IMG}" \
      train_dataloader.dataset.ann_file="${TRAIN_ANN}" \
      val_dataloader.dataset.img_folder="${VAL_IMG}" \
      val_dataloader.dataset.ann_file="${VAL_ANN}" \
      train_dataloader.total_batch_size=4 \
      val_dataloader.total_batch_size=4 \
      train_dataloader.num_workers=0 \
      val_dataloader.num_workers=0 \
      epoches=8
)

echo "[2/2] Plotting training curves..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/metrics/plot_rtdetr_res50_curves.py" \
  --log-file "${LOG_FILE}" \
  --output-image "${PLOT_FILE}" \
  --title "RT-DETR Res50 Backbone on coco8"

echo "Done."
echo "Log: ${LOG_FILE}"
echo "Plot: ${PLOT_FILE}"
