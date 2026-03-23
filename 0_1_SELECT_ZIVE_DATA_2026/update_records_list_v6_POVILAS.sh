# chmod +x update_records_list_v6.sh


# +++++++++++++++++++++++  ALGIRDO SERVERIS +++++++++++++++++++++++

python update_records_list_v6.py \
  --excel AtsisiustiDuomenys_2026/KJ_-_2026-03-13_2225_to_2026-03-14_0636_8h_10min_test/KJ_10min_test_records_summary.xlsx \
  --dir AtsisiustiDuomenys_2026/KJ_-_2026-03-13_2225_to_2026-03-14_0636_8h_10min_test/69b525c88534dd1619e9675f/recordings \
  --cfg-denoising "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir /home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET//MODEL_UNET \
  --denoising \
  --cfg-ectopy "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir /home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
  --fs 200 \
#   --out "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \
# To make denoising == False, just do not include --denoising.

# # VISI ANOTUOTI IRASAI
# python update_records_list_v6.py \
#   --excel "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys/visi_zive_irasai_annot_isplestas-Darb.xlsx" \
#   --dir "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
#   --fs 200 \
#   --cfg-denoising "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --unet-model-dir /home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
#   --denoising \
#   --cfg-ectopy "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
#   --ectopy-model-dir /home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
#   # --out "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \
