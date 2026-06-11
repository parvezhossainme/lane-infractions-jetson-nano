#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python virtualenv not found at ${PYTHON_BIN}"
  exit 1
fi

EPOCHS="${EPOCHS:-20}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/backbone_benchmark/res50_coco128_traffic_person_retrain_${EPOCHS}e}"
LOG_FILE="${OUT_DIR}/log.txt"
TRAIN_PLOT_FILE="${OUT_DIR}/rtdetr_res50_training_curves.png"
CONF_PLOT_DIR="${OUT_DIR}/eval/plots_vehicle_person"

CONFIG_PATH="${ROOT_DIR}/rt-detr-backbones/res50/rtdetrv2_res50_6x_coco_eval.yml"
CHECKPOINT_URL="https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth"

TRAIN_IMG="${ROOT_DIR}/datasets/coco128_traffic_person/images/train"
TRAIN_ANN="${ROOT_DIR}/datasets/coco128_traffic_person/annotations/instances_train.json"
VAL_IMG="${ROOT_DIR}/datasets/coco128_traffic_person/images/val"
VAL_ANN="${ROOT_DIR}/datasets/coco128_traffic_person/annotations/instances_val.json"

mkdir -p "${OUT_DIR}"

echo "[1/4] Training RT-DETRv2 Res50 on traffic+person subset for ${EPOCHS} epochs..."
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
      epoches="${EPOCHS}"
)

echo "[2/4] Plotting training curves..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/metrics/plot_rtdetr_res50_curves.py" \
  --log-file "${LOG_FILE}" \
  --output-image "${TRAIN_PLOT_FILE}" \
  --title "RT-DETR Res50 Retrain (Traffic+Person)"

echo "[3/4] Generating class-wise precision-confidence and recall-confidence curves..."
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/metrics/plot_coco_eval_curves.py" \
  --eval-pth "${OUT_DIR}/eval/latest.pth" \
  --out-dir "${CONF_PLOT_DIR}" \
  --classes person bicycle car motorcycle bus train truck "traffic light" "stop sign"

echo "[4/4] Running test-only evaluation to export confusion matrix artifacts..."
(
  cd "${ROOT_DIR}/RT-DETR/rtdetrv2_pytorch"
  "${PYTHON_BIN}" tools/train.py \
    -c "${CONFIG_PATH}" \
    -t "${CHECKPOINT_URL}" \
    -d cpu \
    --seed 0 \
    --test-only \
    --output-dir "${OUT_DIR}" \
    -u \
      val_dataloader.dataset.img_folder="${VAL_IMG}" \
      val_dataloader.dataset.ann_file="${VAL_ANN}" \
      val_dataloader.total_batch_size=4 \
      val_dataloader.num_workers=0
)

echo "Done. Outputs:"
echo "- Log: ${LOG_FILE}"
echo "- Training curves: ${TRAIN_PLOT_FILE}"
echo "- Confidence curves: ${CONF_PLOT_DIR}"
echo "- Confusion matrix: ${OUT_DIR}/eval/confusion_matrix.png"
