# chmod +x train_ecg_denoising_unet7.sh

# Navigate to the root of the project ZIVE_2025/TRIUKSMU_DETEKTAVIMAS
# Pastaba: Triukšmui naudojami parametrai, kurie skripte priskirti CONFIG

# Čia standardiniai parametrai, kurie buvo naudojami treniruojant modelį.
python train_ecg_denoising_unet7.py\
  --ecg_records_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_FOR_TRAINING"\
  --model_output "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/TRAIN_UNET/results/resunet_ecg_1024_0_5_0_7_test.keras"\
  --vis_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/TRAIN_UNET/visualisations"\
  --train_split 0.8\
  --log_level WARNING\

# Čia  parametrai testavimui
# python train_ecg_denoising_unet7.py\
#   --ecg_records_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/TRAIN_UNET/ecg_selected_for_test"\
#   --model_output "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/TRAIN_UNET/results/resunet_ecg_1024_0_5_3_7_test.keras"\
#   --vis_dir "$HOME/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/TRAIN_UNET/visualisations"\
#   --train_split 0.8\
#   --log_level WARNING\

# INFO messages going to file and console
# DEBUG messages going to file only

# log_dir = 'logs'

# resunet_ecg_1024_0_5_3_6.keras:
# 1024 - 'SEGMENT_LENGTH'
# 0_5 - 'OVERLAP'
# 3 - papildymas, leidžiantis susigaudyti su kokiais duomenimis buvo mokymas. 3 reiškia, kad naudojama mokymo imtis po 3-io
#  papildymo duomenimis (žr. 3-as etapas - duomenys mokymui). Jei 0, reiškia, kad mokyta iš ecg_selected_for_test
# 7 - pridėjau, kad būtų aišku, kad mokyta su train_ecg_denoising_unet7.py 

