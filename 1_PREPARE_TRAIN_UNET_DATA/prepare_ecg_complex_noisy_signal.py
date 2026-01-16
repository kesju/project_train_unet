# Nuskaitome švarų ECG signalus, pridėjome triukšmą, kad gautume sudėtingesnius signalus.
# Šis kodas naudoja bandpass filtrą, kad pašalintų triukšmą iš švarių signalų, o taip pat
# normalizuoja signalą prieš pridėdamas triukšmą, kad būtų galima kontroliuoti triukšmo lygį.
# Naudojame argparse, kad galėtume lengvai keisti parametrus iš komandinės eilutės.

import os
import numpy as np
from glob import glob
import argparse
import logging
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, low=0.5, high=40, fs=200, order=2):
    b, a = butter(order, [low, high], btype='bandpass', fs=fs)
    return filtfilt(b, a, signal)

def normalize(signal):
    """Normalize signal to zero mean and unit variance."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        print("Signal has near-zero variance; returning zeros")
        return np.zeros_like(signal)
    return (signal - mean) / std


# ---------------- CONFIG ----------------
DEFAULT_CONFIG = {
    'FS': 200,
    'GAUSSIAN_STD_RATIO': 0.05,
    'BASELINE_FREQ': 0.33,
    'BASELINE_AMP': 0.1,
    'POWERLINE_FREQ': 50.0,
    'POWERLINE_AMP': 0.05,
    'MOTION_PROB': 0.05,
    'MOTION_AMP': 1.0,
    'MOTION_DURATION_MIN': 10,
    'MOTION_DURATION_MAX': 50
}


# --gaussian_std_ratio 0.05
# --baseline_freq 0.33
# --baseline_amp 0.1\
# --powerline_freq 50.0
# --powerline_amp 0.002\
# --motion_prob 0.05
# --motion_amp 0.5\


# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------- NOISE GENERATION ----------------
def add_ecg_noise(signal, fs=DEFAULT_CONFIG['FS'], 
                  gaussian_std_ratio=DEFAULT_CONFIG['GAUSSIAN_STD_RATIO'],
                  baseline_freq=DEFAULT_CONFIG['BASELINE_FREQ'],
                  baseline_amp=DEFAULT_CONFIG['BASELINE_AMP'],
                  powerline_freq=DEFAULT_CONFIG['POWERLINE_FREQ'],
                  powerline_amp=DEFAULT_CONFIG['POWERLINE_AMP'],
                  motion_prob=DEFAULT_CONFIG['MOTION_PROB'],
                  motion_amp=DEFAULT_CONFIG['MOTION_AMP'],
                  motion_duration_min=DEFAULT_CONFIG['MOTION_DURATION_MIN'],
                  motion_duration_max=DEFAULT_CONFIG['MOTION_DURATION_MAX']):
    """Add Gaussian noise, baseline wander, powerline interference, and motion artifacts to a signal."""
    noisy_signal = signal.copy()
    
    noisy_signal = normalize(noisy_signal)
    
    # 1. Gaussian Noise
    if gaussian_std_ratio > 0:
        gaussian_noise = np.random.normal(0, gaussian_std_ratio, signal.shape)
        noisy_signal += gaussian_noise
        logger.debug(f"Added Gaussian noise: std={gaussian_std_ratio}")

    # 2. Baseline Wander
    if baseline_amp > 0:
        t = np.arange(len(signal)) / fs
        phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander = baseline_amp * np.sin(2 * np.pi * baseline_freq * t + phase)
        noisy_signal += baseline_wander
        logger.debug(f"Added baseline wander: freq={baseline_freq}, amp={baseline_amp}")

    # 3. Powerline Interference
    if powerline_amp > 0:
        t = np.arange(len(signal)) / fs
        phase = np.random.uniform(0, 2 * np.pi)
        powerline_noise = powerline_amp * np.sin(2 * np.pi * powerline_freq * t + phase)
        noisy_signal += powerline_noise
        logger.debug(f"Added powerline interference: freq={powerline_freq}, amp={powerline_amp}")

    # 4. Motion Artifacts
    if motion_amp > 0 and motion_prob > 0:
        motion_signal = np.zeros_like(signal)
        num_samples = len(signal)
        for i in range(num_samples):
            if np.random.random() < motion_prob:
                duration = np.random.randint(motion_duration_min, motion_duration_max + 1)
                if i + duration < num_samples:
                    t_spike = np.arange(duration)
                    spike = motion_amp * np.exp(-((t_spike - duration / 2) ** 2) / (duration / 4) ** 2)
                    motion_signal[i:i + duration] += spike
        noisy_signal += motion_signal
        logger.debug(f"Added motion artifacts: prob={motion_prob}, amp={motion_amp}")

    return noisy_signal

# ---------------- MAIN ----------------
def generate_noisy_ecg(clean_dir, noisy_dir, **noise_params):
    """Generate noisy ECG files from clean ECG files."""
    
    print(f"Generating noisy ECG files from {clean_dir} to {noisy_dir}...")
    
    os.makedirs(noisy_dir, exist_ok=True)
    clean_files = sorted(glob(os.path.join(clean_dir, '*.npy')))
    if not clean_files:
        raise ValueError(f"No .npy files found in {clean_dir}")

    print(f"Found {len(clean_files)} clean files in {clean_dir}. Generating noisy files...")
    print(clean_files)
    print("\n")

    for clean_path in clean_files:
        try:
            clean = np.load(clean_path)
            if clean.size == 0 or clean.ndim != 1:
                logger.warning(f"Invalid or empty file: {clean_path}")
                continue
            
            clean = bandpass_filter(clean)
            
            # adding noise to the clean ECG signal
            noisy = add_ecg_noise(clean, fs=DEFAULT_CONFIG['FS'], **noise_params)
            
            # writing the noisy ECG signal to a new file
            base_name = os.path.splitext(os.path.basename(clean_path))[0]  # '1002_6'
            noisy_filename = f"{base_name}_noised.npy"
            noisy_path = os.path.join(noisy_dir, noisy_filename)
            # noisy_path = os.path.join(noisy_dir, os.path.basename(clean_path).replace('clean', 'noisy'))
            print(os.path.basename(clean_path))
            print(f"Saving noisy file to {noisy_path}")
            np.save(noisy_path, noisy.astype(np.float32))
            logger.info(f"Generated noisy file: {noisy_path}")
            
            # reading accompaning json file
            json_path = os.path.splitext(clean_path)[0] + '.json'
            if os.path.exists(json_path):
                with open(json_path, 'r') as json_file:
                    json_data = json_file.read()
                    
                # writing the json file to the new file
                noisy_filename = f"{base_name}_noised.json"
                noisy_json_path = os.path.join(noisy_dir, noisy_filename)
                # noisy_json_path = os.path.splitext(noisy_path)[0] + '.json'
                with open(noisy_json_path, 'w') as noisy_json_file:
                    noisy_json_file.write(json_data)
                logger.info(f"Copied JSON file: {noisy_json_path}")
            else:
                logger.warning(f"JSON file not found for {clean_path}")
        except FileNotFoundError:
            logger.error(f"File not found: {clean_path}")
            continue
    
        except Exception as e:
            logger.error(f"Failed to process {clean_path}: {str(e)}")
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate noisy ECG files from clean ECG files.')
    parser.add_argument('--clean_dir', type=str, required=True, help='Directory with clean ECG .npy files')
    parser.add_argument('--noisy_dir', type=str, required=True, help='Output directory for noisy ECG .npy files')
    parser.add_argument('--gaussian_std_ratio', type=float, default=DEFAULT_CONFIG['GAUSSIAN_STD_RATIO'], 
                        help='Standard deviation of Gaussian noise')
    parser.add_argument('--baseline_freq', type=float, default=DEFAULT_CONFIG['BASELINE_FREQ'], 
                        help='Frequency of baseline wander (Hz)')
    parser.add_argument('--baseline_amp', type=float, default=DEFAULT_CONFIG['BASELINE_AMP'], 
                        help='Amplitude of baseline wander')
    parser.add_argument('--powerline_freq', type=float, default=DEFAULT_CONFIG['POWERLINE_FREQ'], 
                        help='Frequency of powerline interference (Hz)')
    parser.add_argument('--powerline_amp', type=float, default=DEFAULT_CONFIG['POWERLINE_AMP'], 
                        help='Amplitude of powerline interference')
    parser.add_argument('--motion_prob', type=float, default=DEFAULT_CONFIG['MOTION_PROB'], 
                        help='Probability of motion artifact per sample')
    parser.add_argument('--motion_amp', type=float, default=DEFAULT_CONFIG['MOTION_AMP'], 
                        help='Amplitude of motion artifacts')
    parser.add_argument('--motion_duration_min', type=int, default=DEFAULT_CONFIG['MOTION_DURATION_MIN'], 
                        help='Minimum duration of motion artifacts (samples)')
    parser.add_argument('--motion_duration_max', type=int, default=DEFAULT_CONFIG['MOTION_DURATION_MAX'], 
                        help='Maximum duration of motion artifacts (samples)')
    args = parser.parse_args()

    noise_params = {
        'gaussian_std_ratio': args.gaussian_std_ratio,
        'baseline_freq': args.baseline_freq,
        'baseline_amp': args.baseline_amp,
        'powerline_freq': args.powerline_freq,
        'powerline_amp': args.powerline_amp,
        'motion_prob': args.motion_prob,
        'motion_amp': args.motion_amp,
        'motion_duration_min': args.motion_duration_min,
        'motion_duration_max': args.motion_duration_max
    }

    generate_noisy_ecg(args.clean_dir, args.noisy_dir, **noise_params)