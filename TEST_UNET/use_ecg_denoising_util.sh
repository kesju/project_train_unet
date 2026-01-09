# chmod +x use_ecg_denoising_util.sh


python use_ecg_denoising_util.py \
  --dir "$HOME/DI/2025_ZIVEO/S-ITP-25-9/DATA/ZIVE_DATA/ecg_npy_all" \
  --filename "1005_10.npy"

# 24. 1005_10.npy | outliers:3 rdropouts:0 kiti triukšmai:2 | triukšmų dalis: 15.16%
# quality:0 noni:nan tag:None mark:nan N:586 S:1 V:0 comment:Didžioji dalis švari, tik pasitaiko staigių BW ir keletas EMA
# file_name = "1005_10.npy"