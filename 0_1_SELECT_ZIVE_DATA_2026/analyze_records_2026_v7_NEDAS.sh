#!/usr/bin/env bash

# chmod +x analyze_records_2026_v7_NEDAS.sh

# NEDAS

BASE_DIR="/Users/kesju/DI/DUOMENYS/DUOMENYS_NAUJI_2026"
RECORD_ID="6578733c255f84dd175430de"

DATA_DIR="$BASE_DIR/$RECORD_ID"

conda run -n ITP259 python analyze_records_2026_v7.py \
  --dir "$DATA_DIR" \
  --fs 200 \
  --cfg-denoising "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/denoising_config.yaml" \
  --unet-model-dir "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_UNET" \
  --disable-motions \
  --cfg-ectopy "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_VU_CNN" \
  --out "$DATA_DIR/${RECORD_ID}_records_summary.xlsx"
