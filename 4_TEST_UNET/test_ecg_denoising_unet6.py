# Improved version of your test_saved_model script with enhanced logging, structure, and comments

# https://grok.com/chat/f29dbc93-bc18-48c6-b284-51e9f77c9b04
# https://chatgpt.com/c/68264b89-9abc-8002-9981-26e36a7247a4


# Add  Synthetic Noise: If noisy data isn’t available,
# add an option to generate synthetic noise for testing.

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

# Improved version of your test_saved_model script with enhanced logging, structure, and comments

import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import argparse
import shutil
import tensorflow as tf
import keras
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import logging
import shutil
import neurokit2 as nk

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------- LOGGING ----------------
def setup_logging(debug=False, log_path='test_saved_model.log'):
    class MatplotlibPillowFilter(logging.Filter):
        def filter(self, record):
            if record.name.startswith('matplotlib') or 'findfont' in record.msg.lower():
                return False
            if record.name.startswith('PIL') and 'STREAM' in record.msg:
                return False
            return True

    level = logging.DEBUG if debug else logging.INFO
    root = logging.getLogger()
    while root.handlers:
        root.removeHandler(root.handlers[0])

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode='w')
        ]
    )

    if debug:
        logging.getLogger('').addFilter(MatplotlibPillowFilter())
        logging.getLogger('matplotlib').setLevel(logging.INFO)
        logging.getLogger('PIL').setLevel(logging.INFO)

    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)  # Placeholder; initialized later in main

# ---------------- CONFIG ----------------
DEFAULT_CONFIG = {
    'FS': 200,
    'SEGMENT_LENGTH': 1024,
    'OVERLAP': 0.5,
    'BATCH_SIZE': 32,
    'GAUSSIAN_STD_RATIO': 0.05,
    'BASELINE_FREQ': 0.33,
    'BASELINE_AMP': 0.1,
    'POWERLINE_FREQ': 50.0,
    'POWERLINE_AMP': 0.05,
    'MOTION_PROB': 0.002,
    'MOTION_AMP': 0.5,
    'MOTION_DURATION_MIN': 10,
    'MOTION_DURATION_MAX': 50
}

# DEFAULT_CONFIG = {
#     'FS': 200,
#     'SEGMENT_LENGTH': 1024,
#     'OVERLAP': 0.5,
#     'BATCH_SIZE': 32,
#     'GAUSSIAN_STD_RATIO': 0.05,
#     'BASELINE_FREQ': 0.33,
#     'BASELINE_AMP': 0.1,
#     'POWERLINE_FREQ': 50.0,
#     'POWERLINE_AMP': 0.05,
#     'MOTION_PROB': 0.002,
#     'MOTION_AMP': 0.5,
#     'MOTION_DURATION_MIN': 10,
#     'MOTION_DURATION_MAX': 50
# }

# ---------------- UTILS ----------------
def normalize(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

# bandpass_filter
def bandpass_filter(signal, fs=200, lowcut=0.5, highcut=90., method="butterworth", order=5):
    filtered = nk.signal_filter(signal, fs, lowcut, highcut, method, order)
    return np.asarray(filtered, dtype=float)

def print_mean_std(ecg_signal):
            # Calculate mean and standard deviation
            mean_val = np.mean(ecg_signal)
            std_val = np.std(ecg_signal)
            print(f"mean: {mean_val:.4f} std: {std_val:.4f}")
        
# def add_ecg_noise(signal, fs, gaussian_std_ratio, baseline_freq, baseline_amp,
#                   powerline_freq, powerline_amp, motion_prob, motion_amp,
#                   motion_duration_min, motion_duration_max):
def add_ecg_noise(signal, **kwargs):
    fs = kwargs.pop('fs', DEFAULT_CONFIG['FS'])
    gaussian_std_ratio = kwargs.pop('gaussian_std_ratio', DEFAULT_CONFIG['GAUSSIAN_STD_RATIO'])
    baseline_freq = kwargs.pop('baseline_freq', DEFAULT_CONFIG['BASELINE_FREQ'])
    baseline_amp = kwargs.pop('baseline_amp', DEFAULT_CONFIG['BASELINE_AMP'])
    powerline_freq = kwargs.pop('powerline_freq', DEFAULT_CONFIG['POWERLINE_FREQ'])
    powerline_amp = kwargs.pop('powerline_amp', DEFAULT_CONFIG['POWERLINE_AMP'])
    print("\nGaussian noise, baseline, powerline parameters:", gaussian_std_ratio, baseline_freq, baseline_amp, powerline_freq, powerline_amp)
    motion_prob = kwargs.pop('motion_prob', DEFAULT_CONFIG['MOTION_PROB'])
    motion_amp = kwargs.pop('motion_amp', DEFAULT_CONFIG['MOTION_AMP'])
    motion_duration_min = kwargs.pop('motion_duration_min', DEFAULT_CONFIG['MOTION_DURATION_MIN'])
    motion_duration_max = kwargs.pop('motion_duration_max', DEFAULT_CONFIG['MOTION_DURATION_MAX'])
    print("\nMotion parameters:", motion_prob, motion_amp, motion_duration_min, motion_duration_max)
    
    noisy = signal.astype(np.float64).copy()
    logger.debug(f"Adding noise to signal with shape {signal.shape}")
    
    noisy = normalize(noisy)
    # print(f"\nNormalized signal filtrated: {noisy[:10]}...")
    
    if gaussian_std_ratio > 0:
        noise = np.random.normal(0, gaussian_std_ratio, signal.shape)
        noisy += noise
    if baseline_amp > 0:
        t = np.arange(len(signal)) / fs
        baseline = baseline_amp * np.sin(2 * np.pi * baseline_freq * t + np.random.rand())
        noisy += baseline
    if powerline_amp > 0:
        t = np.arange(len(signal)) / fs
        power = powerline_amp * np.sin(2 * np.pi * powerline_freq * t + np.random.rand())
        noisy += power
    if motion_amp > 0 and motion_prob > 0:
            # 'With Gaussian (Smooth) Motion Noise'
        # motion = np.zeros_like(signal, dtype=np.float64)
        # for i in range(len(signal)):
        #     if np.random.rand() < motion_prob:
        #         d = np.random.randint(motion_duration_min, motion_duration_max + 1)
        #         if i + d < len(signal):
        #             spike = motion_amp * np.exp(-((np.arange(d) - d/2)**2) / (d/4)**2)
        #             motion[i:i+d] += spike
            # With Step (Abrupt) Motion Artifact
        i = 0
        while i < len(noisy):
            if np.random.rand() < motion_prob:
                duration = min(np.random.randint(motion_duration_min, motion_duration_max + 1), len(noisy) - i)
                noisy[i:i+duration] += motion_amp * (np.random.rand() - 0.5)
                i += duration
            else:
                i += 1
    # print(f"\nNoisy signal after adding noise to normalised signal: {noisy[:10]}...")
    return noisy

def segment_signal(signal, segment_length, overlap):
    if len(signal) < segment_length:
        return []
    step = int(segment_length * (1 - overlap))
    return [signal[i:i + segment_length] for i in range(0, len(signal) - segment_length + 1, step)]

def load_ecg_npy_files(directory, desc="Loading ECG files"):
    paths = sorted(glob(os.path.join(directory, '*.npy')))
    print(f"Found {paths} .npy files in {directory}")
    if not paths:
        raise ValueError(f"No .npy files found in {directory}")
    ecgs = []
    for path in tqdm(paths, desc=desc):
        try:
            ecg = np.load(path)
            if isinstance(ecg, np.ndarray) and ecg.ndim == 1 and ecg.size > 0:
                ecgs.append(ecg)
            else:
                logger.warning(f"Invalid ECG shape in {path}: {ecg.shape if isinstance(ecg, np.ndarray) else 'not ndarray'}")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
    if not ecgs:
        raise ValueError(f"No valid ECGs loaded from {directory}")
    return ecgs

def prepare_dataset_generator(clean_ecgs, noisy_ecgs=None, apply_filter=True, synthetic_noise=False, **kwargs):
    segment_length = kwargs.pop('segment_length', DEFAULT_CONFIG['SEGMENT_LENGTH'])
    overlap = kwargs.pop('overlap', DEFAULT_CONFIG['OVERLAP'])
    fs = kwargs.pop('fs', DEFAULT_CONFIG['FS'])

    for idx, clean in enumerate(clean_ecgs):
        # print("\nclean:", clean[:10])
        
        clean_filt = bandpass_filter(clean) if apply_filter else clean
        # print("\nclean_filt:", clean_filt[:10])
        print("\nsynthetic_noise:", synthetic_noise)
        
        if synthetic_noise:
            noisy_filt = add_ecg_noise(clean_filt, fs=fs, **kwargs) # prideda triukšmą prie normalizuoto signalo
        else:
            noisy_filt = bandpass_filter(noisy_ecgs[idx]) if noisy_ecgs else clean_filt

        clean_segs = segment_signal(clean_filt, segment_length, overlap)
        noisy_segs = segment_signal(noisy_filt, segment_length, overlap)
        
        test_clean_seg = clean_segs[0]
        # print("\nclean_seg_filtered[0]:", test_clean_seg[:10])
        test_clean_seg = normalize(test_clean_seg)
        # print("test_clean_seg normalized:", test_clean_seg[:10])
        print_mean_std(test_clean_seg)
        
        test_noisy_seg = noisy_segs[0]
        # print("\nnoisy_segs_filtered[0]:", test_noisy_seg[:10])
        test_noisy_seg = normalize(test_noisy_seg)
        # print("test_noisy_seg normalized:", test_noisy_seg[:10])
        
        print_mean_std(test_noisy_seg)

        for c, n in zip(clean_segs, noisy_segs):
            yield normalize(n)[:, np.newaxis], normalize(c)[:, np.newaxis]

# ---------------- METRICS ----------------
def compute_mse(clean, denoised):
    return np.mean((clean - denoised) ** 2, axis=1)

def compute_snr(clean, denoised):
    signal_power = np.mean(clean ** 2, axis=1)
    noise_power = np.mean((clean - denoised) ** 2, axis=1)
    return 10 * np.log10(signal_power / (noise_power + 1e-8))

def compute_corr(clean, denoised):
    return np.array([pearsonr(c, d)[0] if np.std(c) > 1e-8 and np.std(d) > 1e-8 else 0.0 for c, d in zip(clean, denoised)])

def detect_noise(clean, noisy, mse_thresh=0.15, snr_thresh=10.0, corr_thresh=0.85):
    print("\nDetecting noise in segments...")
    print(f"mse_thresh: {mse_thresh}, snr_thresh: {snr_thresh}, corr_thresh: {corr_thresh}")
    mse = compute_mse(clean, noisy)
    snr = compute_snr(clean, noisy)
    corr = compute_corr(clean, noisy)
    flags = (mse > mse_thresh) | (snr < snr_thresh) | (corr < corr_thresh)
    
    return flags, mse, snr, corr


#
# ---------------- MAIN ----------------
def main(args):
    global logger
    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, "test_saved_model.log")    
    
    logger = setup_logging(debug=args.debug, log_path=log_path)
    logger.info("Starting ECG denoising test")
    model = keras.models.load_model(args.model_path, compile=False)
    if model is None:
        logger.error(f"Failed to load model from {args.model_path}")
        raise RuntimeError(f"Failed to load model from {args.model_path}")
    logger.info(f"Loaded model from {args.model_path}")

    clean_ecgs = load_ecg_npy_files(args.clean_dir, desc="Loading clean ECGs")
    print(f"Loaded {len(clean_ecgs)} clean ECGs from {args.clean_dir}")
    
    noisy_ecgs = load_ecg_npy_files(args.noisy_dir, desc="Loading noisy ECGs") if args.noisy_dir else None
    print(f"Loaded {len(noisy_ecgs)} noisy ECGs from {args.noisy_dir}" if noisy_ecgs else "No noisy ECGs provided")
    
    gen_args = {
        'fs': DEFAULT_CONFIG['FS'],
        'gaussian_std_ratio': args.gaussian_std_ratio,
        'baseline_freq': args.baseline_freq,
        'baseline_amp': args.baseline_amp,
        'powerline_freq': args.powerline_freq,
        'powerline_amp': args.powerline_amp,
        'motion_prob': args.motion_prob,
        'motion_amp': args.motion_amp,
        'motion_duration_min': args.motion_duration_min,
        'motion_duration_max': args.motion_duration_max,
        'segment_length': DEFAULT_CONFIG['SEGMENT_LENGTH'],
        'overlap': DEFAULT_CONFIG['OVERLAP'],
    }

    print(f"\ngen_args: {gen_args}")

    dataset = tf.data.Dataset.from_generator(
        lambda: prepare_dataset_generator(clean_ecgs, noisy_ecgs, apply_filter=not args.no_filter,
                                            synthetic_noise=args.synthetic_noise, **gen_args),
        output_signature=(
            tf.TensorSpec(shape=(DEFAULT_CONFIG['SEGMENT_LENGTH'], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(DEFAULT_CONFIG['SEGMENT_LENGTH'], 1), dtype=tf.float32)
        )
    ).batch(DEFAULT_CONFIG['BATCH_SIZE'])
    
    print(f"\nDataset created with {DEFAULT_CONFIG['BATCH_SIZE']} batch size")  

    logger.info("Predicting with model")
    X_test, Y_test, denoised_output = [], [], []
    for X_batch, Y_batch in tqdm(dataset):
        Y_test.append(Y_batch.numpy())  # Collect clean segments
        X_test.append(X_batch.numpy()) # Collect noisy segments
        denoised_output.append(model.predict(X_batch, verbose=0)) # Predict denoised segments from noisy segments

    X_test = np.concatenate(X_test)
    Y_test = np.concatenate(Y_test)
    denoised_output = np.concatenate(denoised_output)

    # print("\nX_test.squeeze().shape:", X_test.squeeze().shape)
    # print("\nY_test.squeeze().shape:", Y_test.squeeze().shape)
    # print("\ndenosed_output.squeeze().shape:", denoised_output.squeeze().shape)

    mse = compute_mse(Y_test.squeeze(), denoised_output.squeeze())
    snr = compute_snr(Y_test.squeeze(), denoised_output.squeeze())
    corr = compute_corr(Y_test.squeeze(), denoised_output.squeeze())

    logger.info(f"Mean MSE: {np.mean(mse):.6f}")
    logger.info(f"Mean SNR: {np.mean(snr):.2f} dB")
    logger.info(f"Mean Corr: {np.mean(corr):.3f}")

    # Noise detection: Flag noisy segments based on input (X_test) vs. clean (Y_test)
    noisy_flags, mse_list, snr_list, corr_list = detect_noise(Y_test.squeeze(), X_test.squeeze(),  mse_thresh=0.15)
    # def detect_noise(clean, noisy, mse_thresh=0.15, snr_thresh=10.0, corr_thresh=0.85):
    logger.info(f"Detected {np.sum(noisy_flags)} noisy segments out of {len(noisy_flags)}")
    # for idx, flag in enumerate(noisy_flags):
    #     if flag:
    #         logger.debug(f"Noisy segment {idx}: MSE={mse_list[idx]:.4f}, SNR={snr_list[idx]:.2f} dB, Corr={corr_list[idx]:.3f}")

    visualisations_dir = 'visualisations'
    output_dir = os.path.join(visualisations_dir, "denoising_plots")

    # If the directory exists, delete it
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Create a new empty directory
    os.makedirs(output_dir)

    # Plot most extreme noisy segments (by descending MSE)
    noisy_indices = np.where(noisy_flags)[0]
    top_extreme_indices = noisy_indices[np.argsort(mse_list[noisy_indices])[::-1]]
    print("\n")
    print("\nSorted Noisy Segments (by MSE, descending):")
    print("Nr    |Index  | MSE    | SNR (dB) | Corr")
    print("------|-------|--------|----------|-------")
    for i, idx in enumerate(top_extreme_indices):
        print(f"{i+1:5d} | {idx:5d} | {mse_list[idx]:.4f} | {snr_list[idx]:>8.2f} | {corr_list[idx]:>5.3f}")

    for i, idx in enumerate(top_extreme_indices[:args.max_plots]):
    # for idx in range(len(noisy_flags)):
        fig, axs = plt.subplots(3, 1, figsize=(12, 8))
        axs[0].plot(Y_test[idx].squeeze())
        axs[0].set_title(f"Clean (Segment {idx})")
        axs[1].plot(X_test[idx].squeeze(), color='orange')
        axs[1].set_title("Noisy")
        axs[2].plot(denoised_output[idx].squeeze(), color='green')
        axs[2].set_title("Denoised")
        for ax in axs: ax.grid(True)
        fig.suptitle(f"Segment {idx} — MSE: {mse_list[idx]:.4f}, SNR: {snr_list[idx]:.2f} dB, Corr: {corr_list[idx]:.3f}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f"extreme_noisy_segment_{idx:04d}.png"))
        plt.close(fig)

    logger.info(f"Saved {min(len(noisy_indices), args.max_plots)} most extreme noisy segment plots to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--clean_dir', required=True)
    parser.add_argument('--noisy_dir')
    parser.add_argument('--max_plots', type=int, default=10)
    parser.add_argument('--no_filter', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--synthetic_noise', action='store_true')
    parser.add_argument('--gaussian_std_ratio', type=float, default=DEFAULT_CONFIG['GAUSSIAN_STD_RATIO'])
    parser.add_argument('--baseline_freq', type=float, default=DEFAULT_CONFIG['BASELINE_FREQ'])
    parser.add_argument('--baseline_amp', type=float, default=DEFAULT_CONFIG['BASELINE_AMP'])
    parser.add_argument('--powerline_freq', type=float, default=DEFAULT_CONFIG['POWERLINE_FREQ'])
    parser.add_argument('--powerline_amp', type=float, default=DEFAULT_CONFIG['POWERLINE_AMP'])
    parser.add_argument('--motion_prob', type=float, default=DEFAULT_CONFIG['MOTION_PROB'])
    parser.add_argument('--motion_amp', type=float, default=DEFAULT_CONFIG['MOTION_AMP'])
    parser.add_argument('--motion_duration_min', type=int, default=DEFAULT_CONFIG['MOTION_DURATION_MIN'])
    parser.add_argument('--motion_duration_max', type=int, default=DEFAULT_CONFIG['MOTION_DURATION_MAX'])
    args = parser.parse_args()
    main(args)
