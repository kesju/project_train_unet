# chmod +x evaluation_of_accuracy_of_ectopic_beats_detection_GUNDAS.sh

# +++++++++++++++++++++++  GUNDAS +++++++++++++++++++++++

# # Testavimui trumpam
# python evaluation_of_accuracy_of_ectopic_beats_detection_v4.py \
#   --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test" \
#   --exclude-list "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test/exclude_list.txt" \
#   --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --unet-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET//MODEL_UNET \
#   --denoising \
#   --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
#   --ectopy-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
#   --fs 200 \

# --disable-motions
# --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
# --quiet \ - kol kas nenaudojamas
# --all-records \
# action="store_true",
# help="Process all records. By default only first 5 records are processed.",

# Galimi 3 variantai:
# 1) --denoising (įjungia visą denoising pipeline, įskaitant motions etapą, t.y. bus atliekamas triukšmo mažinimas
#     ir judesių artefktų šalinimo operacijos)
# 2) --denoising + --disable-motions (visiškai išjungia motions etapą, t.y. nebus atliekama jokių judesių artefktų 
#     šalinimo operacijų, bet bus atliekamas triukšmo mažinimas)
# 3) Nėra --denoising (visiškai išjungia denoising pipeline, t.y. nebus atliekamas nei triukšmo mažinimas,
#     nei judesių artefktų šalinimo operacijos)  


# Testavimui ilgam
python evaluation_of_accuracy_of_ectopic_beats_detection_v4.py \
  --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
  --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  --unet-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET//MODEL_UNET \
  --denoising \
  --disable-motions \
  --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
  --fs 200 \
  --quiet \

  # --exclude-list "/home/kestutis/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test/exclude_list.txt" \



# VISI ANOTUOTI IRASAI
# python evaluation_of_accuracy_of_ectopic_beats_detection.py \
#   --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
#   --exclude_files "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/ecg_selected_for_test/exclude_list.txt" \
#   --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
#   --unet-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_UNET \
#   # --denoising \
# --cfg-ectopy "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/ectopy_config.yaml" \
# --ectopy-model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/MODEL_VU_CNN \
# --fs 200 \
#   # --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui_added_ml_noise.xlsx"  
#   # --quiet \

