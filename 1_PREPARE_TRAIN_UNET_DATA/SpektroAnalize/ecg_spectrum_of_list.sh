
# chmod +x ecg_spectrum_of_list.sh

src="/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/DATA_ORIG/ecg_zive_npy"

python ecg_spectrum_of_list.py \
  --source "$src" \
  --files 1009_0.npy 1009_1.npy 1010_0.npy \
  --fs 200 \
  --baseline highpass --hp-cutoff 0.5 \
  --save-dir "$src/_spec_plots" \
  --no-show