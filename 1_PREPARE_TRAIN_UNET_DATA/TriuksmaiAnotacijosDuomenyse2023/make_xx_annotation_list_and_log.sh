# chmod +x make_xx_annotation_list_and_log.sh

python make_xx_annotation_list_and_log.py \
  --list-txt 06_Sarasas_anotavimui_testas.txt \
  --src-xlsx /home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/1_PREPARE_TRAIN_UNET_DATA/ScriptsForAnalysis/visi_zive_irasai_atrankai._modif_v1.xlsx \
  --template-xlsx template_anotavimo_logas.xlsx \
  --out-xlsx 06_Anotavimo_logas.xlsx