#!/usr/bin/env bash
set -euo pipefail

export TF_CPP_MIN_LOG_LEVEL=2

RESULTS_DIR="/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/results"
mkdir -p "$RESULTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/evaluation_POVILAS_${STAMP}.log"
SUMMARY_FILE="$RESULTS_DIR/evaluation_POVILAS_${STAMP}_summary.txt"

python evaluation_of_accuracy_of_ectopic_beats_detection_v4_ready_with_summary.py \
  --dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
  --exclude-list "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys/exclude_list.txt" \
  --cfg-denoising "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET" \
  --denoising \
  --disable-motions \
  --cfg-ectopy "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN" \
  --fs 200 \
  --all-records \
  --quiet \
  --summary-out "$SUMMARY_FILE" \
  > 2>&1 | tee "$LOG_FILE"

echo "Done."
echo "Full log    : $LOG_FILE"
echo "Clean summary: $SUMMARY_FILE"
