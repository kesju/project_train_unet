# chmod +x train_ecg_denoising_unet7.sh

# Navigate to the root of the project ZIVE_2025/TRIUKSMU_DETEKTAVIMAS
# Pastaba: Triukšmui naudojami parametrai, kurie yra CONFIG

# Čia standardiniai parametrai, kurie buvo naudojami treniruojant modelį.
# python train_ecg_denoising_unet6.py\
#   --ecg_records_dir "$HOME/DI/2025_ZIVEO/DUOMENYS_UPD/train_test_ecg_mixed"\
#   --model_output "$HOME/DI/2025_ZIVEO/DUOMENYS_UPD/PARAMETRAI/resunet_ecg_1024_0_5_3_7.keras"\
#   --vis_dir "$HOME/DI/2025_ZIVEO/visualisations"\
#   --train_split 0.8\

# Čia  parametrai testavimui
python train_ecg_denoising_unet7.py\
  --ecg_records_dir "$HOME/DI/2025_ZIVEO/S-ITP-25-9/DATA/ZIVE_DATA/ecg_selected_for_test"\
  --model_output "$HOME/DI/2025_ZIVEO/S-ITP-25-9/MODEL_UNET/resunet_ecg_1024_0_5_0_7.keras"\
  --vis_dir "$HOME/DI/2025_ZIVEO/S-ITP-25-9/TRAIN_UNET/visualisations"\
  --train_split 0.8\
  --log_level WARNING\

# INFO messages going to file and console
# DEBUG messages going to file only

# log_dir = 'TRIUKSMU_DETEKTAVIMAS/logs'

# resunet_ecg_1024_0_5_3_6.keras:
# 1024 - 'SEGMENT_LENGTH'
# 0_5 - 'OVERLAP'
# 3 - papildymas, leidžiantis susigaudyti su kokiais duomenimis buvo mokymas. 3 reiškia, kad naudojama mokymo imtis po 3-io
#  papildymo duomenimis (žr. 3-as etapas - duomenys mokymui). Jei 0, reiškia, kad mokyta iš ecg_selected_for_test
# 7 - pridėjau, kad būtų aišku, kad mokyta su train_ecg_denoising_unet7.py 

