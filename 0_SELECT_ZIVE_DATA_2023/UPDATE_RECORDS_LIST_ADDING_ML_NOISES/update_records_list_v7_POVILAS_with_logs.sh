#!/usr/bin/env bash
set -euo pipefail

# This script updates the Excel records list and writes a full console log.
# chmod +x update_records_list_v7_POVILAS_with_logs.sh

export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

RESULTS_DIR="/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/results"
mkdir -p "$RESULTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/update_records_list_v7_${STAMP}.log"

python -u update_records_list_v7_with_logs.py \
  --excel "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys/visi_zive_irasai_annot-Darb.xlsx" \
  --dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_04_14" \
  --fs 200 \
  --cfg-denoising "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET" \
  --denoising \
  --cfg-ectopy "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN" \
  2>&1 | tee "$LOG_FILE"


echo "Done."
echo "Full log: $LOG_FILE"
