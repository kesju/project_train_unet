# chmod +x analyze_records_2026_v5.sh

#!/usr/bin/env bash
set -euo pipefail

# python analyze_records_2026_v5.py \
#   "1_5_Anotacijos_logai/recordings_5.zip" \
#   --fs 200 \
#   --out "1_5_Anotacijos_logai/records_summary_recordings_5-5.xlsx"


# Testavimui
python analyze_records_2026_v5.py \
  "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Anotuoti_visi_duomenys" \
  --fs 200 \
  --out "records_summary_2026_v5_visi_atsisiusti.csv"

