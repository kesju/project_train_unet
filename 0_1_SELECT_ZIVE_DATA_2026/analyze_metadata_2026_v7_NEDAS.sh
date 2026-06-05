#!/usr/bin/env bash

# chmod +x analyze_metadata_2026_v7_NEDAS.sh

# NEDAS

BASE_DIR="/Users/kesju/DI/DUOMENYS/DUOMENYS_NAUJI_2026"
RECORD_ID="/6578733c255f84dd175430de-228"

DATA_DIR="$BASE_DIR/$RECORD_ID"

conda run -n ITP259 python analyze_metadata_2026_v7.py \
  --dir "$DATA_DIR" \
  --fs 200 \
  --out "$DATA_DIR/${RECORD_ID}_metadata_summary.xlsx"
