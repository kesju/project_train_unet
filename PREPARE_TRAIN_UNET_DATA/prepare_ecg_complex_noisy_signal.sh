# chmod +x prepare_ecg_complex_noisy_signal.sh

# Navigate to the root of the project ZIVE_2025/TRIUKSMU_DETEKTAVIMAS


# Run the training script with arguments
python prepare_ecg_complex_noisy_signal.py\
  --clean_dir "$HOME/DI/ZIVEO_2025/DUOMENYS_UPD/clean_ecg_for_test"\
  --noisy_dir "$HOME/DI/ZIVEO_2025/DUOMENYS_UPD/noisy_ecg_for_test"