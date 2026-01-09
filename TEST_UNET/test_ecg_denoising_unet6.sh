# Add Gaussian noise, baseline wander, powerline interference, and motion artifacts to a signal.
# chmod +x train_ecg_denoising_unet7.sh

# https://grok.com/chat/f29dbc93-bc18-48c6-b284-51e9f77c9b04

# To match your model’s training, use the exact noise_params values from training. 
# If unavailable, the defaults are reasonable but may need adjustment:


# Gaussian Noise (gaussian_std_ratio):
# Default: 0.1. Increase (e.g., 0.2) for stronger noise, decrease (e.g., 0.05) for subtler noise.
# Baseline Wander (baseline_freq, baseline_amp):
# Default: 0.3 Hz, 0.05. Adjust frequency (e.g., 0.1–0.5 Hz) or amplitude (e.g., 0.1) per training.
# Powerline Interference (powerline_freq, powerline_amp):
# Default: 60 Hz, 0.05. Use 50 Hz if training was in a 50 Hz region. Adjust amplitude as needed.
# Motion Artifacts (motion_prob, motion_amp, motion_duration_min/max):
# Default: 0.01, 0.2, 10–50 samples. Tune probability (e.g., 0.005) or 
# amplitude (e.g., 0.5) to match training artifact frequency and intensity.

# Test the model with synthetic noise or pre-generated noisy data.
# Option 1: Synthetic Noise:
# python test_ecg_denoising_unet6..py --model_path model.keras --clean_dir data/clean --synthetic_noise \
#     --gaussian_std_ratio 0.1 --baseline_freq 0.3 --baseline_amp 0.05 \
#     --powerline_freq 60.0 --powerline_amp 0.05 \
#     --motion_prob 0.01 --motion_amp 0.2 \
#     --max_plots 5 --debug

# chmod +x test_ecg_denoising_unet6.sh

# Navigate to the root of the project ZIVE_2025/TRIUKSMU_DETEKTAVIMAS
# log yra

# https://grok.com/chat/f29dbc93-bc18-48c6-b284-51e9f77c9b04


# Option 2: Pre-generate Noisy Data:
# python test_ecg_denoising_unet6.py --clean_dir data/clean --noisy_dir data/noisy \
#     --gaussian_std_ratio 0.1 --baseline_freq 0.3 --baseline_amp 0.05 \
#     --powerline_freq 50.0 --powerline_amp 0.05 \
#     --motion_prob 0.01 --motion_amp 0.2
# python test_saved_model.py --model_path model.keras --clean_dir data/clean --noisy_dir data/noisy \
#     --max_plots 5 --debug

# Alternatyvos:
  # --model_path "$HOME/DI/2005_ZIVEO/trained_models/resunet_ecg.keras"\
  # --clean_dir "$HOME/DI/2005_ZIVEO/S-ITP-25-9/clean_ecg_for_test"\


# python test_ecg_denoising_unet6.py\
#   --model_path "$HOME/DI/2005_ZIVEO/S-ITP-25-9/MODEL_UNET/resunet_ecg_3.keras"\
#   --clean_dir "$HOME/DI/2005_ZIVEO/S-ITP-25-9/clean_ecg_for_test"\
#   --synthetic_noise\
#   --gaussian_std_ratio 0.05 --baseline_freq 0.33 --baseline_amp 0.1\
#   --powerline_freq 50.0 --powerline_amp 0.05\
#   --motion_prob 0.002 --motion_amp 0.5\
#   --motion_duration_min 10 --motion_duration_max 50\
#   --max_plots 20\
#   --debug

python test_ecg_denoising_unet6.py \
  --model_path "$HOME/DI/2025_ZIVEO/S-ITP-25-9/MODEL_UNET/resunet_ecg_1024_0_5_3_7.keras" \
  --clean_dir "$HOME/DI/2025_ZIVEO/S-ITP-25-9/DATA/clean_ecg_for_test" \
  --synthetic_noise \
  --gaussian_std_ratio 0.1 \
  --baseline_freq 0.33 --baseline_amp 0.2 \
  --powerline_freq 50.0 --powerline_amp 0.1 \
  --motion_prob 0.01 --motion_amp 1.0 \
  --motion_duration_min 20 --motion_duration_max 100 \
  --max_plots 20 \
  --debug



# Recommended parameters range for synthetic noise generation:
# noise_args = {
#     'gaussian_std_ratio': np.random.uniform(0.03, 0.15),
#     'baseline_freq': 0.33,
#     'baseline_amp': np.random.uniform(0.05, 0.25),
#     'powerline_freq': 50.0,
#     'powerline_amp': np.random.uniform(0.05, 0.15),
#     'motion_prob': np.random.uniform(0.002, 0.015),
#     'motion_amp': np.random.uniform(0.3, 1.2),
#     'motion_duration_min': 10,
#     'motion_duration_max': 100,
#     'fs': 200



# It loads the model, evaluates MSE/SNR/correlation, and plots a denoised ECG segment.
# log_path = 'TESTAVIMUI/test_saved_model.log'
# plotting = 'TESTAVIMUI/denoising_plots_xxxx.../'
    
# Įrašai aplanke clean_ecg_for_test turi būti švarūs, be triukšmo (json neturi būti anotuotų triukšmų indeksų) 

# resunet_ecg_3.keras gautas su:
# 'SEGMENT_LENGTH': 1024,
# 'OVERLAP': 0.5,

# 'NOISE_PARAMS': {
#         'gaussian_std_ratio': 0.05,
#         'baseline_freq': 0.33,
#         'baseline_amp': 0.1,
#         'powerline_freq': 50,
#         'powerline_amp': 0.05,
#         'motion_prob': 0.002,
#         'motion_amp': 0.5
#     }

#  --motion_prob 0.1 --motion_amp 1.0\