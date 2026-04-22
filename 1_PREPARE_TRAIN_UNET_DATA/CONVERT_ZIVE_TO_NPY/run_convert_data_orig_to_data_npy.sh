#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="convert_data_orig_to_data_npy.py"

DATA_ORIG="/home/kesju/DI/2025_ZIVEO/DUOMENYS_ANOTUOTI/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_04_14"
DATA_NPY="data_npy"
CONVERSION_TABLE="ConversionTable.xlsx"
LIST_FILE="basenames_to_convert.txt"
OUTPUT_DIR="results/convert_data_orig_to_data_npy"

python "$PYTHON_SCRIPT" \
  --data-orig "$DATA_ORIG" \
  --data-npy "$DATA_NPY" \
  --conversion-table "$CONVERSION_TABLE" \
  --list-file "$LIST_FILE" \
  --output-dir "$OUTPUT_DIR"

# To overwrite existing converted files, use this instead:
# python "$PYTHON_SCRIPT" \
#   --data-orig "$DATA_ORIG" \
#   --data-npy "$DATA_NPY" \
#   --conversion-table "$CONVERSION_TABLE" \
#   --list-file "$LIST_FILE" \
#   --output-dir "$OUTPUT_DIR" \
#   --overwrite
