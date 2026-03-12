# chmod +x train_ecg_denoising_unet8.sh

# Čia standardiniai parametrai, kurie buvo naudojami treniruojant modelį.
python train_ecg_denoising_unet8.py\
  --ecg_records_dir "DATA_FOR_TRAINING"\
  --ecg_list_file "Irasai_modeliui_1024_0_5_0_8_test.txt"\
  --model_output "resunet_ecg_1024_0_5_0_8_test.keras"\
  --vis_dir "visualisations"\
  --train_split 0.8\
  --log_level WARNING\

