# chmod +x evaluation_of_accuracy_of_ectopic_beats_detection_POVILAS.sh

# +++++++++++++++++++++++  POVILAS +++++++++++++++++++++++

# # Testavimui
python evaluation_of_accuracy_of_ectopic_beats_detection.py \
  --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
  --exclude-list "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test/exclude_list.txt" \
  --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET//MODEL_UNET \
  --denoising \
  --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
  --fs 200 \
#   --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \
# To make denoising == False, just do not include --denoising.


# VISI ANOTUOTI IRASAI
# python evaluation_of_accuracy_of_ectopic_beats_detection.py \
#   --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
#   --exclude_files "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys/exclude_list.txt" \
#   --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --unet-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
#   # --denoising \
# --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
# --ectopy-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
# --fs 200 \
#   # --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \

