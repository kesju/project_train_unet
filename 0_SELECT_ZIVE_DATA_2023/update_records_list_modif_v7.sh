# chmod +x update_records_list_modif_v7.sh

# python update_records_list_modif_v7.py \
#   --excel "visi_zive_irasai_atrankai._modif_v1 - Darb.xlsx" \
#   --zip "Anotuoti_ir_atrinkti_duomenys" \
#    --fs 200 \
#   --out "visi_zive_irasai_atrankai._modif_v1 - Darb_updated.xlsx" 

python update_records_list_modif_v7_updated.py \
  --excel "visi_zive_irasai_atrankai._modif_v1 - Darb.xlsx" \
  --zip "AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys" \
   --fs 200 \
  --out "visi_zive_irasai_atrankai._modif_v1 - Darb_updated.xlsx"   

# # Testavimui
#   python update_records_list_modif_v7_updated.py \
#   --excel "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb.xlsx" \
#   --zip "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui" \
#    --fs 200 \
#   --out "/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/DuomenysTestui/visi_zive_irasai_atrankai._modif_v1 - Darb_updated.xlsx"   