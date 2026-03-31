#!/usr/bin/env bash
set -euo pipefail

# This script evaluates the accuracy of ectopic beat detection using the GUNDAS method, 
# with logs and a summary output.
# chmod +x evaluation_of_accuracy_of_ectopic_beats_detection_GUNDAS_with_logs.sh

export TF_CPP_MIN_LOG_LEVEL=2

RESULTS_DIR="/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/results"
mkdir -p "$RESULTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/evaluation_GUNDAS_${STAMP}.log"
SUMMARY_FILE="$RESULTS_DIR/evaluation_GUNDAS_${STAMP}_summary.txt"

python evaluation_of_accuracy_of_ectopic_beats_detection_v4_ready_with_summary.py \
  --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
  --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET" \
  --denoising \
  --disable-motions \
  --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN" \
  --fs 200 \
  --quiet \
  --summary-out "$SUMMARY_FILE"
  > "$LOG_FILE" 2>&1

echo "Done."
echo "Full log    : $LOG_FILE"
echo "Clean summary: $SUMMARY_FILE"

  # --all-records \