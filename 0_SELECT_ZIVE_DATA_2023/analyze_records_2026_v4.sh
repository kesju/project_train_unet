# chmod +x analyze_records_2026_v4.sh

#!/usr/bin/env bash
set -euo pipefail

python analyze_records_2026_v4.py \
  "1_5_Anotacijos_logai/recordings_5.zip" \
  --fs 200 \
  --out "1_5_Anotacijos_logai/records_summary_recordings_5-4.xlsx"


# Testavimui
# python analyze_records_2026_v4.py \
#   "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_COLLECT_ZIVE_DATA/DUOM_2026_01_27/Igno/4_records_igno_2026" \
#   --fs 200 \
#   --out "records_summary_2026_v4_IGNO_4.csv"

