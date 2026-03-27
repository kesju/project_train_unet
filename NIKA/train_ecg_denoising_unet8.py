"""
Train a 1D U-Net model for ECG denoising using ECG records stored as `.npy` files
with accompanying noise annotations in `.json` files.

Main purpose
------------
The script prepares a training dataset from ECG records located in `ecg_records_dir`,
builds noisy/clean training pairs, trains the ECG denoising U-Net model, and saves
the trained model and related outputs.

Input data
----------
The script expects ECG records in a directory, where each selected ECG signal file:
- is stored as a `.npy` file
- has a corresponding `.json` file with noise interval annotations

Record selection
----------------
The script supports two modes of selecting ECG records:

1. Default mode:
   If `--ecg_list_file` is not provided, or if the provided list file is empty,
   the script uses all non-empty ECG records found in `ecg_records_dir`.

2. Filtered mode:
   If `--ecg_list_file` is provided and contains filenames, only the ECG records
   listed in that file are used for training.

The list file should contain one ECG filename per line. The script accepts either
plain filenames or paths; in both cases only the filename itself is used for matching
against files in `ecg_records_dir`.

Additional validation for `--ecg_list_file`
-------------------------------------------
When `--ecg_list_file` is provided and is not empty, the script first validates the list
before training starts. For each listed ECG file, it checks that:
- the `.npy` ECG file exists in `ecg_records_dir`
- the corresponding `.json` annotation file also exists in `ecg_records_dir`

If any listed ECG file or JSON file is missing, the script stops with an error before
loading data or starting training.

Loaded data
-----------
For each valid selected record, the script loads:
- ECG signal
- noise indices / annotated noisy intervals
- filename
- patient ID

These data are then used to create the training dataset for the denoising model.

Typical usage
-------------
- Use all records from directory:
    python train_ecg_denoising_unet7.py --ecg_records_dir /path/to/data ...

- Use only records listed in a text file:
    python train_ecg_denoising_unet7.py \
        --ecg_records_dir /path/to/data \
        --ecg_list_file /path/to/list.txt ...

Notes
-----
- Empty ECG records are skipped.
- The script preserves the old behavior when no list file is supplied.
- The filename list mechanism allows reproducible training on a predefined subset
  of ECG records.
"""

# Prideda visų rušių triukšmus, kuriuos galima nurodyti per NOISE_LEVELS
# NOISE_LEVELS = ["clean", "mild", "moderate", "severe"]
# https://chatgpt.com/c/6849ae9b-79e4-8002-b09b-47b0545f64ae

# train_and_test_model:
# 
# # 1. Load all ECG records and noise indices from the specified directory.
# # 2. Split the records into training and test sets based on patient IDs.
# # 3. Count valid segments in both training and test sets.
# # 4. Create TensorFlow datasets for training and testing.
        # create_tf_dataset(clean_ecgs, noise_indices_list,...
            # for ecg, noise_indices in zip(clean_ecgs, noise_indices_list):
            # ecg = lowcut_filter(ecg, fs=CONFIG['FS'])
            # clean_segments = segment_signal(ecg, CONFIG['SEGMENT_LENGTH'], CONFIG['OVERLAP'], noise_indices, target_fs=CONFIG['FS'])
            # for seg in clean_segments:
            #     seg = normalize(seg)
            #     noisy_seg = normalize(apply_noise(seg, CONFIG['NOISE_PARAMS']))
            #     yield noisy_seg[..., np.newaxis], seg[..., np.newaxis]   

# # 5. Build a ResUNet-based Denoising Autoencoder model.
# # 6. Train the model with early stopping and model checkpointing.
# # 7. Evaluate the model on the test set and visualize results.
# # 8. Save the trained model and visualizations of denoised examples.
# # 9. Log all steps and metrics for reproducibility and debugging.
# # 10. Handle exceptions and edge cases gracefully, logging errors and warnings.
# # 11. Ensure reproducibility by setting random seeds for NumPy and TensorFlow.

import os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from scipy.stats import pearsonr
import argparse
import logging
import json
from datetime import datetime
import uuid
import neurokit2 as nk
from sklearn.model_selection import train_test_split
import time
import re

# === Išoriniai moduliai (lygiagretus aplankas) =================================
PARALLEL_PATH = Path().resolve().parent / "SUPL_FUNCTIONS"
sys.path.append(str(PARALLEL_PATH))

# Pakeitimai, kuriuos reikia padaryti
order = 4

# lowcut_filter
def lowcut_filter(signal, fs=200, lowcut=0.5, method="butterworth", order=4):
    filtered = nk.signal_filter(
        signal,
        sampling_rate=fs,
        lowcut=lowcut,
        highcut=None,
        method=method,
        order=order
    )
    return np.asarray(filtered, dtype=float)

# Suppress TensorFlow debug messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create log directory
PROJECT_SUBDIR = Path(__file__).resolve().parent   # .../2_TRAIN_UNET
log_dir = PROJECT_SUBDIR / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Create log file with timestamp
log_file = os.path.join(log_dir, f'ecg_denoising_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

logger.info(f"\n\n\nStarting ECG denoising script. Log file: {log_file}")
logger.info(f"TensorFlow version: {tf.__version__}")

# ---------------- CONFIG ----------------
CONFIG = {
    'FS': 200,
    'SEGMENT_LENGTH': 1024,
    'OVERLAP': 0.5,
    'BATCH_SIZE': 32,
    'EPOCHS': 30,
    'NOISE_PARAMS': {
        'gaussian_std_ratio': 0.15,
        'baseline_freq': 0.33,
        'baseline_amp': 0.25,
        'powerline_freq': 50,
        'powerline_amp': 0.15,
        'motion_prob': 0.015,
        'motion_amp': 1.2,
        'motion_duration_min': 10,
        'motion_duration_max': 100
    }
}

def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("    " * indent + f"{key}:")
            print_dict(value, indent + 1)
        else:
            print("    " * indent + f"{key}: {value}")


# training script used these parameters:
# 'gaussian_std_ratio': np.random.uniform(0.03, 0.15),      # mild to strong Gaussian
# 'baseline_amp': np.random.uniform(0.05, 0.25),            # mild to strong baseline
# 'baseline_freq': 0.33,                                    # can randomize if you want
# 'powerline_amp': np.random.uniform(0.05, 0.15),           # mild to strong powerline
# 'powerline_freq': 50.0,                                   # can be 60.0 if needed
# 'motion_amp': np.random.uniform(0.3, 1.2),                # mild to strong artifact
# 'motion_prob': np.random.uniform(0.002, 0.015),           # rare to common artifact
# 'motion_duration_min': 10,
# 'motion_duration_max': 100

#   --synthetic_noise \
#   --gaussian_std_ratio 0.1 \
#   --baseline_freq 0.33 --baseline_amp 0.2 \
#   --powerline_freq 50.0 --powerline_amp 0.1 \
#   --motion_prob 0.01 --motion_amp 1.0 \
#   --motion_duration_min 20 --motion_duration_max 100 \

# ---------------- DATA LOADING ----------------
def get_ecg_noise_indices_annotated(json_path):
    """Load annotated noise indices from JSON file."""
    try:
        with open(json_path, 'r', encoding='UTF-8', errors='ignore') as f:
            data = json.load(f)
        noise_indices_from_json = data.get('noises_annotated', [])
        noise_indices = [(item['startIndex'], item['endIndex']) for item in noise_indices_from_json]
        logger.debug(f"Loaded annotated noise indices from {json_path}: {noise_indices}")
        return noise_indices
    except Exception as e:
        logger.warning(f"Failed to load noise indices from {json_path}: {e}. Assuming no noise regions.")
        return []

def read_requested_filenames(list_file):
    """Read requested ECG filenames from a text file. Empty or missing content means use all files."""
    if list_file is None:
        return []

    list_path = Path(list_file)
    if not list_path.exists():
        raise ValueError(f"ECG list file does not exist: {list_path}")

    with list_path.open('r', encoding='utf-8') as f:
        requested_files = [line.strip() for line in f if line.strip()]

    logger.info(f"Read {len(requested_files)} non-empty filename entries from {list_path}")
    return requested_files


def check_requested_files_and_jsons(directory, requested_files, list_file=None):
    """Validate that each listed ECG file and its accompanying JSON exist in the target directory."""
    directory_path = Path(directory)
    list_path = Path(list_file) if list_file is not None else None

    if not directory_path.exists():
        raise ValueError(f"Directory {directory_path} does not exist")

    existing_files = {p.name for p in directory_path.iterdir() if p.is_file()}

    found_pairs = []
    missing_data_files = []
    missing_json_files = []
    missing_any = []
    report_rows = []

    for fname in requested_files:
        data_name = Path(fname).name
        json_name = Path(fname).with_suffix('.json').name

        data_exists = data_name in existing_files
        json_exists = json_name in existing_files

        if data_exists and json_exists:
            found_pairs.append(data_name)
        else:
            missing_any.append(data_name)
            if not data_exists:
                missing_data_files.append(data_name)
            if not json_exists:
                missing_json_files.append(json_name)

        report_rows.append({
            'data_file': data_name,
            'json_file': json_name,
            'data_status': 'OK' if data_exists else 'MISSING',
            'json_status': 'OK' if json_exists else 'MISSING',
        })

    logger.info(f"Folder checked: {directory_path}")
    if list_path is not None:
        logger.info(f"List file: {list_path}")
    logger.info(f"Files in list: {len(requested_files)}")
    logger.info(f"Full pairs found: {len(found_pairs)}")
    logger.info(f"Records with anything missing: {len(missing_any)}")
    logger.info(f"Missing data files: {len(missing_data_files)}")
    logger.info(f"Missing json files: {len(missing_json_files)}")

    for row in report_rows:
        logger.info(
            f"{row['data_file']:<30} data: {row['data_status']:<8} json: {row['json_status']:<8} ({row['json_file']})"
        )

    if missing_data_files or missing_json_files:
        if missing_data_files:
            logger.error("Missing data files: " + ", ".join(missing_data_files))
        if missing_json_files:
            logger.error("Missing accompanying json files: " + ", ".join(missing_json_files))
        raise ValueError(
            "The provided --ecg_list_file contains missing ECG data files and/or accompanying JSON files. "
            "See log output for details."
        )

    logger.info("All listed data files and their accompanying json files exist.")
    return {Path(fname).name for fname in requested_files}


def load_ecg_npy_files(directory, requested_filenames=None):
    """Load .npy ECG files and corresponding noise indices from JSON files, ensuring consistency."""
    logger.info(f"Loading ECG and noise indices from {directory}")
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        raise ValueError(f"Directory {directory} does not exist")

    file_paths = sorted(glob(os.path.join(directory, '*.npy')))
    if not file_paths:
        logger.error(f"No .npy files found in {directory}")
        raise ValueError(f"No .npy files found in {directory}")

    requested_set = None
    if requested_filenames:
        requested_set = {Path(fname).name for fname in requested_filenames}
        logger.info(f"Filtering ECG files using provided list: {len(requested_set)} filenames requested")
        file_paths = [path for path in file_paths if os.path.basename(path) in requested_set]
        logger.info(f"Selected {len(file_paths)} .npy files from {directory} after applying filename list filter")

    ecgs = []
    noise_indices_list = []
    filenames = []
    patient_ids = []
    # Regex to match xxxx_xx.npy or xxxx_x.npy
    filename_pattern = re.compile(r'^(\d{4})_(\d{1,2})\.npy$')

    for path in file_paths:
        fname = os.path.basename(path)
        match = filename_pattern.match(fname)
        if not match:
            logger.warning(f"Filename {fname} does not match expected format xxxx_xx.npy or xxxx_x.npy; skipping")
            continue

        patient_id = match.group(1)  # Extract xxxx
        try:
            ecg = np.load(path)
            if ecg.size == 0:
                logger.warning(f"Empty file: {path}")
                continue

            # Load corresponding JSON file
            json_path = os.path.splitext(path)[0] + '.json'
            noise_indices = get_ecg_noise_indices_annotated(json_path)
            ecgs.append(ecg)
            noise_indices_list.append(noise_indices)
            filenames.append(fname)
            patient_ids.append(patient_id)
            logger.debug(f"Loaded file: {path}, shape: {ecg.shape}, noise indices: {noise_indices}, patient ID: {patient_id}")

        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue

    if not ecgs:
        logger.error(f"No valid ECG files found in {directory}")
        raise ValueError(f"No valid ECG files found in {directory}")

    logger.info(f"Loaded {len(ecgs)} valid .npy files with corresponding noise indices from {directory}")
    return ecgs, noise_indices_list, filenames, patient_ids

# ---------------- PREPROCESSING ----------------
def normalize(signal):
    """Normalize signal to zero mean and unit variance."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        logger.warning("Signal has near-zero variance; returning zeros")
        return np.zeros_like(signal)
    return (signal - mean) / std


# ---------------- NOISE FUNCTIONS ----------------

def sample_random_noise_params():
    """Sample noise parameters for one segment (randomly within realistic ranges)."""
    return {
        'gaussian_std_ratio': np.random.uniform(0.03, 0.15),      # mild to strong Gaussian
        'baseline_freq': 0.33,                                    # can randomize if you want
        'baseline_amp': np.random.uniform(0.05, 0.25),            # mild to strong baseline
        'powerline_freq': 50.0,                                   # can be 60.0 if needed
        'powerline_amp': np.random.uniform(0.05, 0.15),           # mild to strong powerline
        'motion_prob': np.random.uniform(0.002, 0.015),           # rare to common artifact
        'motion_amp': np.random.uniform(0.3, 1.2),                # mild to strong artifact
        'motion_duration_min': 10,
        'motion_duration_max': 100
    }

# training script used these parameters:
# 'gaussian_std_ratio': np.random.uniform(0.03, 0.15),      # mild to strong Gaussian
# 'baseline_amp': np.random.uniform(0.05, 0.25),            # mild to strong baseline
# 'baseline_freq': 0.33,                                    # can randomize if you want
# 'powerline_amp': np.random.uniform(0.05, 0.15),           # mild to strong powerline
# 'powerline_freq': 50.0,                                   # can be 60.0 if needed
# 'motion_amp': np.random.uniform(0.3, 1.2),                # mild to strong artifact
# 'motion_prob': np.random.uniform(0.002, 0.015),           # rare to common artifact
# 'motion_duration_min': 10,
# 'motion_duration_max': 100


def add_gaussian_noise(signal, std_ratio=0.05):
    """Add Gaussian noise to signal."""
    return signal + np.random.normal(0, std_ratio * np.std(signal), size=signal.shape)

def add_baseline_wander(signal, freq=0.33, amp=0.1, fs=200):
    """Add baseline wander to signal."""
    t = np.arange(len(signal)) / fs
    return signal + amp * np.sin(2 * np.pi * freq * t)

def add_powerline(signal, freq=50, amp=0.05, fs=200):
    """Add powerline noise to signal."""
    t = np.arange(len(signal)) / fs
    return signal + amp * np.sin(2 * np.pi * freq * t)

def add_motion_artifact(signal, prob=0.002, amp=0.5, duration_min=10, duration_max=50):
    """Add random-duration step artifacts."""
    artifact = signal.copy()
    i = 0
    while i < len(artifact):
        if np.random.rand() < prob:
            duration = min(np.random.randint(duration_min, duration_max+1), len(artifact) - i)
            artifact[i:i+duration] += amp * (np.random.rand() - 0.5)
            i += duration
        else:
            i += 1
    return artifact

def apply_noise(signal, noise_params):
    """Apply multiple noise types to signal."""
    x = add_gaussian_noise(signal, noise_params['gaussian_std_ratio'])
    x = add_baseline_wander(x, noise_params['baseline_freq'], noise_params['baseline_amp'], CONFIG['FS'])
    x = add_powerline(x, noise_params['powerline_freq'], noise_params['powerline_amp'], CONFIG['FS'])
    x = add_motion_artifact(
        x, 
        prob=noise_params['motion_prob'],
        amp=noise_params['motion_amp'],
        duration_min=noise_params.get('motion_duration_min', 10),
        duration_max=noise_params.get('motion_duration_max', 50)
    )
    return x


# ---------------- SEGMENTATION ----------------
def segment_signal(signal, segment_length=1024, overlap=0.5, noise_indices=None, target_fs=200):
    """Segment signal into overlapping windows, excluding noisy regions and invalid segments."""
    if noise_indices is None:
        noise_indices = []
    
    if len(signal) < segment_length:
        logger.warning(f"Signal too short ({len(signal)} < {segment_length}); skipping")
        return []
    
    step = int(segment_length * (1 - overlap))
    segments = []
    
    for start in range(0, len(signal) - segment_length + 1, step):
        end = start + segment_length
        segment = signal[start:end]
        
        # Check if segment overlaps with any noise region
        overlaps_noise = False
        for noise_start, noise_end in noise_indices:
            if not (end <= noise_start or start >= noise_end):
                overlaps_noise = True
                logger.debug(f"Skipping segment [{start}:{end}] due to overlap with noise [{noise_start}:{noise_end}]")
                break
        if overlaps_noise:
            continue
        
        # Check variance
        variance = np.var(segment)
        if variance < 0.01 or variance > 3:
            logger.debug(f"Skipping segment [{start}:{end}] due to variance {variance:.4f} (outside [0.01, 3])")
            continue
        
        # Apply NeuroKit2 cleaning and QRS detection
        try:
            ecg_clean = nk.ecg_clean(segment, sampling_rate=target_fs)
            r_peaks = nk.ecg_peaks(ecg_clean, sampling_rate=target_fs)[1]['ECG_R_Peaks']
            qrs_count = len(r_peaks)
            min_rpeaks = 0.5 * (segment_length / target_fs)
            if qrs_count < min_rpeaks:
                logger.debug(f"Skipping segment [{start}:{end}] due to low QRS count ({qrs_count} < {min_rpeaks})")
                continue
        except Exception as e:
            logger.debug(f"Skipping segment [{start}:{end}] due to NeuroKit2 processing error: {e}")
            continue
        
        segments.append(segment)
        logger.debug(f"Accepted segment [{start}:{end}], variance={variance:.4f}, QRS count={qrs_count}")
    
    logger.info(f"Segmented signal of length {len(signal)} into {len(segments)} valid segments")
    return segments


# ---------------- DATASET CREATION ----------------
def count_segments(clean_ecgs, noise_indices_list, segment_length=1024, overlap=0.5):
    """Count total number of valid segments across all ECG signals."""
    total_segments = 0
    for ecg, noise_indices in zip(clean_ecgs, noise_indices_list):
        ecg = lowcut_filter(ecg)       
        segments = segment_signal(ecg, segment_length, overlap, noise_indices, target_fs=CONFIG['FS'])
        total_segments += len(segments)
        logger.debug(f"ECG signal length: {len(ecg)}, segments: {len(segments)}")
    logger.info(f"Total segments counted: {total_segments}")
    return total_segments


def create_tf_dataset(clean_ecgs, noise_indices_list, batch_size=32, repeat=True, randomize_noise=False):
    """Create a tf.data.Dataset from ECG signals, with random noise injection if randomize_noise=True."""
    logger.info("Creating TensorFlow dataset")
    def generator():
        for ecg, noise_indices in zip(clean_ecgs, noise_indices_list):
            ecg = lowcut_filter(ecg)       
            clean_segments = segment_signal(ecg, CONFIG['SEGMENT_LENGTH'], CONFIG['OVERLAP'], noise_indices, target_fs=CONFIG['FS'])
            for seg in clean_segments:
                seg = normalize(seg)
                if randomize_noise:
                    # Each segment gets new, random noise!
                    noise_params = sample_random_noise_params()
                else:
                    # Use fixed (default) noise params, e.g. for test/validation
                    noise_params = CONFIG['NOISE_PARAMS']
                noisy_seg = normalize(apply_noise(seg, noise_params))
                yield noisy_seg[..., np.newaxis], seg[..., np.newaxis]

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(CONFIG['SEGMENT_LENGTH'], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(CONFIG['SEGMENT_LENGTH'], 1), dtype=tf.float32)
        )
    )
    dataset = dataset.shuffle(1000, seed=42)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    logger.debug("Dataset created")
    return dataset
    

# ---------------- MODEL ----------------
def residual_conv_block(x, filters, kernel_size=3):
    """Residual convolutional block with skip connection."""
    residual = x
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if residual.shape[-1] != x.shape[-1]:
        residual = layers.Conv1D(filters, 1, padding='same')(x)
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters):
    """Encoder block with residual convolution and max pooling."""
    f = residual_conv_block(x, filters)
    p = layers.MaxPooling1D(2)(f)
    return f, p

def decoder_block(x, skip, filters):
    """Decoder block with upsampling and skip connection."""
    x = layers.UpSampling1D(2)(x)
    x = layers.Concatenate()([x, skip])
    return residual_conv_block(x, filters)

def build_resunet_dae(input_shape=(CONFIG['SEGMENT_LENGTH'], 1)):
    """Build a ResUNet-based Denoising Autoencoder."""
    logger.info("Building ResUNet Denoising Autoencoder")
    inputs = layers.Input(shape=input_shape)
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    b = residual_conv_block(p3, 256)
    d3 = decoder_block(b, s3, 128)
    d2 = decoder_block(d3, s2, 64)
    d1 = decoder_block(d2, s1, 32)
    outputs = layers.Conv1D(1, kernel_size=1, activation='linear')(d1)
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ---------------- METRICS ----------------
def compute_metrics(clean, denoised):
    """Compute MSE, SNR, and Pearson correlation."""
    mse = np.mean((clean - denoised) ** 2)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    try:
        corr = pearsonr(clean, denoised)[0]
    except ValueError:
        corr = 0.0
    return mse, snr, corr

# ---------------- VISUALIZATION ----------------
def visualize_example(noisy, clean, denoised, output_dir):
    """Visualize clean, noisy, and denoised signals in separate subplots with metrics."""
    logger.debug(f"Saving visualization to {output_dir}")
    
    # Compute metrics
    mse, snr, corr = compute_metrics(clean, denoised)
    
    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    
    # Plot clean ECG
    ax1.plot(clean, color='blue', label='Clean ECG')
    ax1.set_title('Clean ECG')
    ax1.grid(True)
    ax1.legend()
    
    # Plot noisy ECG
    ax2.plot(noisy, color='orange', label='Noisy ECG')
    ax2.set_title('Noisy ECG')
    ax2.grid(True)
    ax2.legend()
    
    # Plot denoised ECG with metrics
    ax3.plot(denoised, color='green', label='Denoised ECG')
    ax3.set_title('Denoised ECG')
    ax3.grid(True)
    ax3.legend()
    
    # Add metrics as text in the denoised subplot
    metrics_text = f'MSE: {mse:.6f}\nSNR: {snr:.2f} dB\nCorr: {corr:.3f}'
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set overall title and labels
    fig.suptitle('ECG Denoising Comparison', fontsize=16)
    plt.xlabel('Sample Index')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))  # Adjust layout to accommodate suptitle
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    vis_path = os.path.join(output_dir, f'ecg_comparison_{timestamp}_{uuid.uuid4().hex[:8]}.png')
    plt.savefig(vis_path)
    plt.close()
    logger.info(f"Visualization saved: {vis_path}")

# ---------------- MAIN WORKFLOW ----------------
def train_and_test_model(ecg_records_dir, model_output, vis_dir='visualizations', train_split=0.8, ecg_list_file=None):
    """Train and evaluate the ECG denoising model."""
    logger.info(f"Starting training workflow. ECG records dir: {ecg_records_dir}, Model output: {model_output}, Train split: {train_split}")
    os.makedirs(os.path.dirname(model_output) or '.', exist_ok=True)
    
    requested_filenames = read_requested_filenames(ecg_list_file)
    if ecg_list_file and requested_filenames:
        check_requested_filenames = check_requested_files_and_jsons(ecg_records_dir, requested_filenames, ecg_list_file)
    else:
        check_requested_filenames = None
        if ecg_list_file:
            logger.info("The provided --ecg_list_file is empty; using all records found in ecg_records_dir.")

    # Load ECG records, noise indices, filenames, and patient IDs
    ecgs, noise_indices_list, filenames, patient_ids = load_ecg_npy_files(
        ecg_records_dir,
        requested_filenames=check_requested_filenames if check_requested_filenames is not None else requested_filenames,
    )
    
    # Group records by patient ID
    patient_dict = {}
    for ecg, noise_indices, fname, pid in zip(ecgs, noise_indices_list, filenames, patient_ids):
        if pid not in patient_dict:
            patient_dict[pid] = {'ecgs': [], 'noise_indices': [], 'filenames': []}
        patient_dict[pid]['ecgs'].append(ecg)
        patient_dict[pid]['noise_indices'].append(noise_indices)
        patient_dict[pid]['filenames'].append(fname)
    
    # Get unique patient IDs and split them
    unique_pids = list(patient_dict.keys())
    train_pids, test_pids = train_test_split(unique_pids, train_size=train_split, random_state=42)
    
    # Collect training and test data
    train_ecgs = []
    train_noise_indices = []
    train_filenames = []
    test_ecgs = []
    test_noise_indices = []
    test_filenames = []
    
    for pid in train_pids:
        train_ecgs.extend(patient_dict[pid]['ecgs'])
        train_noise_indices.extend(patient_dict[pid]['noise_indices'])
        train_filenames.extend(patient_dict[pid]['filenames'])
    
    for pid in test_pids:
        test_ecgs.extend(patient_dict[pid]['ecgs'])
        test_noise_indices.extend(patient_dict[pid]['noise_indices'])
        test_filenames.extend(patient_dict[pid]['filenames'])
    
    logger.info(f"Split {len(ecgs)} records from {len(unique_pids)} patients into {len(train_ecgs)} training records ({len(train_pids)} patients) and {len(test_ecgs)} test records ({len(test_pids)} patients)")
    
    # Log training and test filenames
    logger.info("Training ECG files:\n" + "\n".join([f"  - {fname}" for fname in train_filenames]))
    logger.info("Test ECG files:\n" + "\n".join([f"  - {fname}" for fname in test_filenames]))
    
    train_segments = count_segments(train_ecgs, train_noise_indices, CONFIG['SEGMENT_LENGTH'], CONFIG['OVERLAP'])
    logger.info(f"Training segments: {train_segments}")
    
    test_segments = count_segments(test_ecgs, test_noise_indices, CONFIG['SEGMENT_LENGTH'], CONFIG['OVERLAP'])
    logger.info(f"Test segments: {test_segments}")
    
    if train_segments == 0:
        logger.error("No valid training segments found")
        raise ValueError("No valid training segments found. Check signal lengths and noise indices.")
    if test_segments == 0:
        logger.error("No valid test segments found")
        raise ValueError("No valid test segments found. Check signal lengths and noise indices.")
    
    logger.info(f"Training segments: {train_segments}, Test segments: {test_segments}")
    
    train_dataset = create_tf_dataset(train_ecgs, train_noise_indices, CONFIG['BATCH_SIZE'], repeat=True, randomize_noise=True)
    test_dataset = create_tf_dataset(test_ecgs, test_noise_indices, CONFIG['BATCH_SIZE'], repeat=False, randomize_noise=False)

    
    steps_per_epoch = max(1, train_segments // CONFIG['BATCH_SIZE'])
    validation_steps = max(1, test_segments // CONFIG['BATCH_SIZE'])
    logger.info(f"Steps per epoch: {steps_per_epoch}, Validation steps: {validation_steps}")
    
    # Validate dataset
    logger.info("Validating dataset")
    try:
        for batch_x, batch_y in train_dataset.take(1):
            logger.info(f"Training batch shapes: X={batch_x.shape}, Y={batch_y.shape}")
        for batch_x, batch_y in test_dataset.take(1):
            logger.info(f"Test batch shapes: X={batch_x.shape}, Y={batch_y.shape}")
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise
    
    model = build_resunet_dae()
    model.summary(print_fn=lambda x: logger.info(x))
    
    callbacks_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(model_output, save_best_only=True),
        callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: logger.info(
            f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, MAE: {logs['mae']:.4f}, "
            f"Val Loss: {logs.get('val_loss', 0):.4f}, Val MAE: {logs.get('val_mae', 0):.4f}"
        ))
    ]
    
    logger.info("Starting model training")
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=CONFIG['EPOCHS'],
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    logger.info(f"Model saved to {model_output}")
    
    mse_list, snr_list, corr_list = [], [], []
    test_dataset_unbatched = test_dataset.unbatch().take(test_segments)
    for i, (noisy, clean) in enumerate(test_dataset_unbatched):
        noisy = noisy[np.newaxis, ...]
        denoised = model.predict(noisy, verbose=0)[0].squeeze()
        clean = clean.numpy().squeeze()
        noisy = noisy.numpy().squeeze()
        mse, snr, corr = compute_metrics(clean, denoised)
        mse_list.append(mse)
        snr_list.append(snr)
        corr_list.append(corr)
        
        if np.random.rand() < 0.1:
            visualize_example(noisy, clean, denoised, vis_dir)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Evaluated {i + 1}/{test_segments} test samples")
    
    logger.info("\n--- Average Metrics over Test Set ---")
    logger.info(f"MSE:  {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}")
    logger.info(f"SNR:  {np.mean(snr_list):.2f} ± {np.std(snr_list):.2f} dB")
    logger.info(f"Corr: {np.mean(corr_list):.3f} ± {np.std(corr_list):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D U-Net Denoising Autoencoder for ECG")
    parser.add_argument('--ecg_records_dir', type=str, required=True, help='Path to directory with ECG .npy files')
    parser.add_argument('--model_output', type=str, default='ecg_denoising_model.keras', help='Path to save trained model')
    parser.add_argument('--ecg_list_file', type=str, default=None, help='Optional path to a text file with ECG filenames to use; if omitted or empty, all ECG records are used')
    parser.add_argument('--vis_dir', type=str, default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data to use for training (0.0 to 1.0)')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    args = parser.parse_args()
    
    print("\n========================================")
    print("TRAINING AUTOENCODER UNET7 FOR ECG DENOISING")
    
    print(f"\nConfiguration parameters from train_ecg_denoising_unet7.sh:\n")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
        
    print(f"\nConfiguration parameters from CONFIG:\n")
    print_dict(CONFIG)
    print("========================================\n")

    if not 0.0 < args.train_split < 1.0:
        logger.error("train_split must be between 0.0 and 1.0")
        raise ValueError("train_split must be between 0.0 and 1.0")
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    logger.info(f"Arguments: {vars(args)}")
    
    # Measure execution time of train_and_test_model
    start_time = time.time()
    train_and_test_model(args.ecg_records_dir, args.model_output, args.vis_dir, args.train_split, args.ecg_list_file)
    end_time = time.time()
    duration = end_time - start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info(f"train_and_test_model execution time: {int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}")
