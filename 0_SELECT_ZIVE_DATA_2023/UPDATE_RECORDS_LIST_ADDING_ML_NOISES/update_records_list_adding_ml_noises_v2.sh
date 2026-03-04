# chmod +x update_records_list_adding_ml_noises_v2.sh

# python update_records_list_adding_ml_noises_v1.py \
#   --excel "visi_zive_irasai_atrankai._modif_v1 - Darb.xlsx" \
#   --zip "Anotuoti_ir_atrinkti_duomenys" \
#    --fs 200 \
#   --out "visi_zive_irasai_atrankai._modif_v1 - Darb_updated_added_ml_noise.xlsx" 

# Testavimui
# python update_records_list_adding_ml_noises_v2.py \
#   --excel "AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb - testui.xlsx" \
#   --dir "AtsisiuntimasZiveDuomenu/DuomenysTestui" \
#    --fs 200 \
#   --out "AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb - testui_updated_added_ml_noise.xlsx"   

python update_records_list_adding_ml_noises_v2.py \
  --excel "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/zive_irasai_testui.xlsx" \
  --dir "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
  --fs 200 \
  # --cfg-denoising "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/CONFIG/denoising_config.yaml" \
  # --model-dir /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET//MODEL_UNET \

# # Testavimui
  # python update_records_list_adding_ml_noises_v1.py \
  # --excel "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb.xlsx" \
  # --zip "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
  #  --fs 200 \
  # --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb_updated_added_ml_noise.xlsx"  