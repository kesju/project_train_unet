# chmod +x analyze_records_2026_v6.sh

# 

# python update_records_list_v4.py \
#   --excel "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/PseudoAnnotated/Pseudo_annotated.xlsx" \
#   --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/PseudoAnnotated" \
#   --fs 200 \
#   --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
#   # --denoising \
#   # --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \

# python analyze_records_2026_v6.py \
#   --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
#   --fs 200 \
#   --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
#   # --denoising \
#   # --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \


python analyze_records_2026_v6_1.py \
  --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
  --dir KJ_-_2026-03-13_2225_to_2026-03-14_0636_8h_10min_test/69b525c88534dd1619e9675f/recordings \
  --fs 200 \
  --out KJ_-_2026-03-13_2225_to_2026-03-14_0636_8h_10min_test_records_summary.xlsx