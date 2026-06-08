#!/usr/bin/env bash
set -euo pipefail

# This script updates the Excel records list and writes a full console log.
# chmod +x update_records_list_v7_NEDAS.sh

export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONUNBUFFERED=1

RESULTS_DIR="/home/kestutis/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/results"
mkdir -p "$RESULTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/update_records_list_v7_${STAMP}.log"

python -u update_records_list_v7.py \
  --excel "/home/kestutis/DI/DUOMENYS/DUOMENYS_ANOTUOTI/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_06_08/visi_zive_irasai_annot-Darb_26_04_16_atrankai.xlsx" \
  --dir "/home/kestutis/DI/DUOMENYS/DUOMENYS_ANOTUOTI/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_06_08" \
  --fs 200 \
  --cfg-denoising "/home/kestutis/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/denoising_config.yaml" \
  --unet-model-dir "/home/kestutis/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_UNET" \
  --denoising \
  --cfg-ectopy "/home/kestutis/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir "/home/kestutis/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_VU_CNN" \
  2>&1 | tee "$LOG_FILE"


echo "Done."
echo "Full log: $LOG_FILE"
