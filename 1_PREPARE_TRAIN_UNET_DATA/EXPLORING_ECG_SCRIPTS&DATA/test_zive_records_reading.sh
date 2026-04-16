#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCRIPT_DIR="$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/1_PREPARE_TRAIN_UNET_DATA/EXPLORING_ECG_SCRIPTS&DATA"
PY_SCRIPT="$SCRIPT_DIR/test_zive_records_reading.py"

DATA_DIR="$HOME/DI/2025_ZIVEO/DUOMENYS_ANOTUOTI/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_04_14"
EXCLUDE_LIST="$HOME/DI/2025_ZIVEO/DUOMENYS_ANOTUOTI/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys_26_04_14/exclude_list.txt"

RESULTS_DIR="$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/1_PREPARE_TRAIN_UNET_DATA/EXPLORING_ECG_SCRIPTS&DATA/results"
mkdir -p "$RESULTS_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/test_zive_records_reading_${STAMP}.log"
SUMMARY_FILE="$RESULTS_DIR/test_zive_records_reading_${STAMP}_summary.txt"

# -----------------------------------------------------------------------------
# Start time
# -----------------------------------------------------------------------------
START_EPOCH="$(date +%s)"
START_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
set +e
{
  echo "================================================================================"
  echo "TEST ZIVE RECORDS READING"
  echo "================================================================================"
  echo "Started               : $START_HUMAN"
  echo "Python script         : $PY_SCRIPT"
  echo "Script dir            : $SCRIPT_DIR"
  echo "Data dir              : $DATA_DIR"
  echo "Exclude list          : $EXCLUDE_LIST"
  echo "Results dir           : $RESULTS_DIR"
  echo "Log file              : $LOG_FILE"
  echo "Summary file          : $SUMMARY_FILE"
  echo "================================================================================"

  python -u "$PY_SCRIPT" \
    --dir "$DATA_DIR" \
    --exclude-list "$EXCLUDE_LIST" \
    --all-records
} 2>&1 | tee "$LOG_FILE"

CMD_STATUS=${PIPESTATUS[0]}
set -e

    # --parallel-path "$SCRIPT_DIR" \

# -----------------------------------------------------------------------------
# End time and duration
# -----------------------------------------------------------------------------
END_EPOCH="$(date +%s)"
END_HUMAN="$(date '+%Y-%m-%d %H:%M:%S')"
ELAPSED_SEC=$((END_EPOCH - START_EPOCH))
ELAPSED_MIN=$(awk "BEGIN {printf \"%.1f\", $ELAPSED_SEC/60}")
ELAPSED_H=$((ELAPSED_SEC / 3600))
ELAPSED_M=$(((ELAPSED_SEC % 3600) / 60))
ELAPSED_S=$((ELAPSED_SEC % 60))
ELAPSED_HHMMSS=$(printf "%02d:%02d:%02d" "$ELAPSED_H" "$ELAPSED_M" "$ELAPSED_S")

# -----------------------------------------------------------------------------
# Write summary
# -----------------------------------------------------------------------------
{
  echo "TEST ZIVE RECORDS READING"
  echo "================================================================================"
  echo "Started               : $START_HUMAN"
  echo "Finished              : $END_HUMAN"
  echo "Elapsed seconds       : $ELAPSED_SEC"
  echo "Elapsed minutes       : $ELAPSED_MIN"
  echo "Elapsed hh:mm:ss      : $ELAPSED_HHMMSS"
  echo "Exit status           : $CMD_STATUS"
  echo "Python script         : $PY_SCRIPT"
  echo "Script dir            : $SCRIPT_DIR"
  echo "Data dir              : $DATA_DIR"
  echo "Exclude list          : $EXCLUDE_LIST"
  echo "Log file              : $LOG_FILE"
  echo "Summary file          : $SUMMARY_FILE"
  echo "================================================================================"
} > "$SUMMARY_FILE"

# -----------------------------------------------------------------------------
# Final console info
# -----------------------------------------------------------------------------
echo
echo "Summary written to: $SUMMARY_FILE"
echo "Full log written to: $LOG_FILE"
echo "Exit status: $CMD_STATUS"

exit "$CMD_STATUS"