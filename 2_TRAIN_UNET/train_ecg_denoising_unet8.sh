# chmod +x train_ecg_denoising_unet8.sh

# Čia standardiniai parametrai, kurie buvo naudojami treniruojant modelį.
python train_ecg_denoising_unet8.py\
  --ecg_records_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_FOR_TRAINING"\
  --ecg_list_file "Irasai_modeliui_1024_0_5_0_8_0_test_XX.txt"\
  --model_output "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/log_results/resunet_ecg_1024_0_5_0_8_test.keras"\
  --vis_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/2_TRAIN_UNET/visualisations"\
  --train_split 0.8\
  --log_level WARNING\

