#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
TRAIN_DIR="${ROOT_DIR}/RT-DETR/rtdetrv2_pytorch"
DATASET_DIR="${ROOT_DIR}/laneguard_s-3"
OUT_DIR="${ROOT_DIR}/outputs/laneguard_local_gpu_100e"
LOG_FILE="${OUT_DIR}/train.log"
CHECKPOINT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
CONFIG_PATH="configs/rtdetr/rtdetr_r18vd_6x_coco.yml"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python virtualenv not found at ${VENV_PYTHON}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

cd "${TRAIN_DIR}"

CUDA_VISIBLE_DEVICES=0 "${VENV_PYTHON}" tools/train.py \
  -c "${CONFIG_PATH}" \
  -d cuda:0 \
  -t "${CHECKPOINT_URL}" \
  --use-amp \
  --seed=0 \
  --output-dir "${OUT_DIR}" \
  -u \
    train_dataloader.dataset.img_folder="${DATASET_DIR}/train" \
    train_dataloader.dataset.ann_file="${DATASET_DIR}/train/_annotations.coco.json" \
    val_dataloader.dataset.img_folder="${DATASET_DIR}/valid" \
    val_dataloader.dataset.ann_file="${DATASET_DIR}/valid/_annotations.coco.json" \
    train_dataloader.total_batch_size=2 \
    val_dataloader.total_batch_size=2 \
    train_dataloader.num_workers=2 \
    val_dataloader.num_workers=2 \
    num_classes=2 \
    remap_mscoco_category=False \
    epoches=100 \
  > "${LOG_FILE}" 2>&1

echo "Training finished. Log: ${LOG_FILE}"
