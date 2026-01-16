import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import h5py, json, math
from pathlib import Path


def zive_read_file_1ch(filename):
    with open(filename, 'rb') as f:  # Use 'rb' to read binary file
        a = np.fromfile(f, dtype=np.dtype('>i4'))  # Read file content as big-endian 4-byte integers
    
    ADCmax = 0x800000
    Vref = 2.5
    b = (a - ADCmax / 2) * 2 * Vref / ADCmax / 3.5 * 1000  # Corrected the calculation by adding multiplication symbol
    ecg_signal = b - np.mean(b)
    
    return ecg_signal
  
def get_ecg_signal(args_fileName):
  # Extract the file extension
  file_extension = os.path.splitext(args_fileName)[1]

  # Check if the extension is .h5py
  if file_extension.lower() == '.h5':
        # print("The file has a .h5 extension.")
        with h5py.File(args_fileName, 'r') as f:
          ecg_signal = f['dataset'][:]
          
  # Check if the extension is three digits (excluding the dot)
  elif len(file_extension) == 4 and file_extension[1:].isdigit():
        # print("The file has a three-digit extension.")
        ecg_signal = zive_read_file_1ch(args_fileName)
        
  elif file_extension.lower() == '.npy':
        # print("The file has a .npy extension.")
        ecg_signal = np.load(args_fileName, mmap_mode='r')
        
  # If neither condition is true
  else:
        ecg_signal = np.array([])
        print("The file does not have a .h5py extension or a three-digit extension or .npy extension.")
   
  return ecg_signal


def get_ecg_noise_indices_annotated(json_path):
    
    with open(json_path, 'r', encoding='UTF-8', errors='ignore') as f:
        data = json.load(f)  # use json.load directly instead of json.loads(f.read())
 
    noise_indices_from_json = data.get('noises_annotated', [])

    noise_indices = [(item['startIndex'], item['endIndex']) for item in noise_indices_from_json]
    return noise_indices


def extract_metadata_from_json(json_path):
    """
    Extracts selected metadata fields from a Zive-format JSON file.

    Returns a dictionary with:
    quality, noni, tag, mark, N, S, V, U, recordingId
    """
    with open(json_path, 'r') as f:
        meta = json.load(f)

    counts = meta.get("rpeakAnnotationCounts", {})
    noni = meta.get("noises_annotated", 'NA')
    if noni != 'NA':
        noni = len(noni)

    return {
        'quality': meta.get('quality', 0),
        'noni': noni,
        'tag': meta.get('tag', 'NA'),
        'mark': meta.get('mark', 'NA'),
        'N': counts.get("N", 0),
        'S': counts.get("S", 0),
        'V': counts.get("V", 0),
        'U': counts.get("U", 0),
        'recordingId': meta.get('recordingId', None)
    }

def convert_seconds_to_hms(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return hours, minutes, seconds


def plot_ecg_with_noises(
    ecg_signal,
    rpeaks_df,
    fs,
    file_name="ECG",
    label_of_view = "Original",
    selected_method=1,
    fragment_duration=20,
    outliers_indices=None,
    rdropouts_indices=None
):
    """
    Visualize ECG signal with R-peaks, extrasystoles, outliers, and r-dropouts.

    Parameters:
        ecg_signal (np.ndarray): ECG signal array.
        rpeaks_df (pd.DataFrame): DataFrame with 'rpeak' and 'pred' columns.
        fs (int): Sampling rate in Hz.
        file_name (str): Name used in the plot title.
        selected_method (int): The detection method used (1–4).
        fragment_duration (float): Duration of each fragment in seconds.
        outliers_indices (list of tuples): List of (start, end) sample index tuples for outliers.
        rdropouts_indices (list of tuples): List of (start, end) sample index tuples for r-dropouts.
    """
    if outliers_indices is None:
        outliers_indices = []
    if rdropouts_indices is None:
        rdropouts_indices = []

    signal_length = len(ecg_signal)
    sampling_rate = fs
    duration_seconds = signal_length / sampling_rate
    time_axis = np.linspace(0, duration_seconds, signal_length)

    fragment_samples = int(fragment_duration * sampling_rate)

    for start in range(0, signal_length, fragment_samples):
        end = min(start + fragment_samples, signal_length)
        fragment_time = time_axis[start:end]
        fragment_signal = ecg_signal[start:end]
        
        # fragment_df = rpeaks_df[(rpeaks_df['rpeak'] >= start) & (rpeaks_df['rpeak'] < end)]
        # rpeak_times = fragment_df['rpeak'] / sampling_rate
        # rpeak_amps = ecg_signal[fragment_df['rpeak']]
        
        plt.figure(figsize=(15, 4))
        plt.plot(fragment_time, fragment_signal, label="_nolegend_")

        # Outlier rectangles
        for (o_start, o_end) in outliers_indices:
            if o_end > start and o_start < end:
                x_start = max(o_start, start) / sampling_rate
                x_end = min(o_end, end) / sampling_rate
                plt.axvspan(x_start, x_end, color='orange', alpha=0.3, label="Outlier Episode")

        # R-dropout rectangles
        for (r_start, r_end) in rdropouts_indices:
            if r_end > start and r_start < end:
                x_start = max(r_start, start) / sampling_rate
                x_end = min(r_end, end) / sampling_rate
                plt.axvspan(x_start, x_end, color='cyan', alpha=0.3, label="R-dropout Episode")

        # Normal R-peaks
        # normal_mask = fragment_df['pred'] == 0
        # plt.plot(rpeak_times[normal_mask], rpeak_amps[normal_mask], "ro", label="_nolegend_", markersize=4)

        # # Extrasystoles
        # extras_mask = fragment_df['pred'] != 0
        # if extras_mask.any():
        #     pred_to_mark = {1: "S", 2: "V", 3: "U", 4: "E"}
        #     sub_df = fragment_df[extras_mask]
        #     r_times = sub_df['rpeak'] / sampling_rate
        #     r_amps = ecg_signal[sub_df['rpeak']]
        #     for x, y, p in zip(r_times, r_amps, sub_df['pred']):
        #         mark = pred_to_mark.get(p, "?")
        #         plt.text(x, y + 0.05, mark, fontsize=12, color='black', ha='center')
        #     plt.plot(r_times, r_amps, "ko", label="_nolegend_", markersize=6)

        # label = f"Extrasystole (Method {selected_method})"
        # plt.title(f"{file_name}  {label_of_view}  (Fragment {start // fragment_samples + 1})  {label}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Amplitude")

        # Legend deduplication
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            plt.legend(by_label.values(), by_label.keys(), loc='upper right')

        plt.grid(True)
        plt.tight_layout()
        plt.show()


def divide_signal_into_fragments(signal, portion_length):
    indices = []
    signal_len = len(signal)
    
    for start in range(0, signal_len, portion_length):
        end = min(start + portion_length, signal_len)  # Ensure the last portion does not exceed the signal length
        indices.append((start, end))
    
    return indices

import matplotlib.pyplot as plt
import numpy as np
import math



def plot_signal_write_plot_R_P(fileName, ecg_signal, fs, num_fragment, plot_signal_from_in_secs, plot_signal_to_in_secs, portion_length_in_secs, 
                                plot_save_dir=None, save_mark=None, recID=None,
                                gap1_indices_secs=[], gap2_indices_secs=[],
                                gap3_indices_secs=[], mark_indices_secs=[],
                                annot_df=None, 
                                rpeak_indices_secs=[], ppeak_indices_secs=[],
                                flag_secs=True):

    plot_signal_from = int(plot_signal_from_in_secs*fs)
    # print(f"\nplot_signal_from: {plot_signal_from_in_secs} sec. ({plot_signal_from})")

    plot_signal_to = int(plot_signal_to_in_secs*fs)
    if (plot_signal_to >= len(ecg_signal)): 
        plot_signal_to = len(ecg_signal) - 1
        # plot_signal_to_in_secs = plot_signal_to/fs
        plot_signal_to_in_secs = math.ceil(plot_signal_to/fs)

    # print(f"plot_signal_to {plot_signal_to_in_secs} sec. ({plot_signal_to})")
    # print(f"The length of all plot: {plot_signal_to_in_secs - plot_signal_from_in_secs} sec.")

    portion_length = math.ceil(portion_length_in_secs*fs)
    portion_length_in_secs = math.ceil(portion_length/fs)
    # print(f"portion_length: {portion_length_in_secs} secs. ({portion_length})")


    # if gap_indices_secs:
    #     gap_indices = [(x * fs, y * fs) for x, y in gap_indices_secs] # indexes
    #     print(f"pause indices: {gap_indices}")
        
    # Assuming ecg_signal, ecg_length, fs, portion_length, plot_signal_from, and plot_signal_to are defined
    # t = np.arange(ecg_length)  # Time vector in seconds
    
    ecg_length = len(ecg_signal)
    nt = np.arange(ecg_length)
    # print(nt[:10])
    t = np.arange(ecg_length) / fs  # Time vector in seconds
    # print(t[:10])
    
    # # Create a directory to save the plots
    # os.makedirs(plot_dir, exist_ok=True)

    # Calculate the total number of full portions
    num_portions = int((plot_signal_to - plot_signal_from) // portion_length)

    # # Plot the ECG signal in portions and save each plot

    # print("\nnum_portions:", num_portions)
    for i in range(num_portions + 1):  # Include the last portion
        start_idx = int(plot_signal_from + i * portion_length)
        end_idx = int(start_idx + portion_length)
        
        # Adjust the end index for the last portion
        if end_idx > plot_signal_to:
            end_idx = int(plot_signal_to)
        # print(f"i: {i}  start_idx: {start_idx} end_idx: {end_idx} ")
        
         # Prepare time and corresponding indices
        time_vals = t[start_idx:end_idx]
        index_vals = np.arange(start_idx, end_idx)

        plt.figure(figsize=(15, 5))
        plt.plot(time_vals, ecg_signal[start_idx:end_idx], label=f"Portion {i+1}")
        if recID is not None:
            plt.title(f"File: {fileName}  RecID: {recID}  Frag: {num_fragment}")
        else:
            plt.title(f"File: {fileName} Fragment: {num_fragment}")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()


        if not annot_df.empty:
            fragment_df = annot_df[(annot_df['rpeak'] >= start_idx) & (annot_df['rpeak'] < end_idx)]
            rpeak_idx = fragment_df['rpeak'] / fs
            rpeak_amps = ecg_signal[fragment_df['rpeak']]
        
            # Normal R-peaks
            normal_mask = fragment_df['annot'] == 0
            plt.plot(rpeak_idx[normal_mask], rpeak_amps[normal_mask], "ro", label="_nolegend_", markersize=4)
        
            # Extrasystoles
            extras_mask = fragment_df['annot'] != 0
            if extras_mask.any():
                pred_to_mark = {1: "S", 2: "V", 3: "U", 4: "E"}
                sub_df = fragment_df[extras_mask]
                r_times = sub_df['rpeak'] / fs
                r_amps = ecg_signal[sub_df['rpeak']]
                for x, y, p in zip(r_times, r_amps, sub_df['annot']):
                    mark = pred_to_mark.get(p, "?")
                    plt.text(x, y + 0.05, mark, fontsize=12, color='black', ha='center')
                plt.plot(r_times, r_amps, "ko", label="_nolegend_", markersize=6)
    
        if not flag_secs:    
            # Replace x-axis tick labels with sample indices
            xticks = plt.xticks()[0]  # Get current tick locations (in time)
            xtick_labels = [str(int(x * fs)) for x in xticks]  # Convert time to sample index
            plt.xticks(xticks, xtick_labels)
            plt.xlabel("Sample Index (mapped to time)")
        else:
            plt.xlabel("Time (seconds)")
                
            
        # Highlight gap1 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap1_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='red', alpha=0.5)
        
        # Highlight gap2 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap2_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='blue', alpha=0.5)

        # Highlight gap3 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap3_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='yellow', alpha=0.7)
  
        # Highlight mark intervals on the plot
        for (pause_start_secs, pause_end_secs) in mark_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='grey', alpha=0.3)
        
        # Highlight rpeaks on the plot
        for rpeak_indice_secs in rpeak_indices_secs:
            if (rpeak_indice_secs > t[start_idx] and rpeak_indice_secs < t[end_idx]):
                # plt.axvline(x=rpeak_indice_secs, color='red', linestyle='--', label='R Peaks')
                rpeak_indice = int(rpeak_indice_secs*fs)
                rpeak_value = ecg_signal[rpeak_indice]
                plt.scatter(rpeak_indice_secs, rpeak_value, color='red')

        # Highlight ppeaks on the plot
        for ppeak_indice_secs in ppeak_indices_secs:
            if (ppeak_indice_secs > t[start_idx] and ppeak_indice_secs < t[end_idx]):
                # plt.axvline(x=rpeak_indice_secs, color='red', linestyle='--', label='R Peaks')
                ppeak_indice = int(ppeak_indice_secs*fs)
                rpeak_value = ecg_signal[ppeak_indice]
                plt.scatter(ppeak_indice_secs, rpeak_value, color='blue')
         
         # Extract the numerical part from the fileName
        base_name = os.path.basename(fileName)
        number_part = os.path.splitext(base_name)
        formatted_number_part = number_part[0] +  number_part[1].replace('.', '_') + '_' + save_mark 
                
        # Check if the directory is writable
        if plot_save_dir is not None:
            # Ensure the directory exists
            os.makedirs(plot_save_dir, exist_ok=True)
            
            # Check if the directory is writable
            if not os.access(plot_save_dir, os.W_OK):
                print(f"Error: The directory '{plot_save_dir}' is not writable.", file=sys.stderr)
            else:
                plot_filename = os.path.join(plot_save_dir, f"{num_fragment}_frag_{formatted_number_part}.png")
                plt.savefig(plot_filename)
        else:
            plt.show()

        plt.close()

        # Break the loop if end index reaches plot_signal_to
        if end_idx == plot_signal_to:
            break



        
def plot_gap_legend(gap1, gap2, gap3):
    # Plot small rectangles and notes in one line
    fig, ax = plt.subplots(figsize=(5, 1))

    ax.bar(x=0, height=0.5, width=0.5, color='red', alpha=0.5)
    ax.text(0.6, 0.25, f'{gap1}', va='center')

    ax.bar(x=2, height=0.5, width=0.5, color='blue', alpha=0.5)
    ax.text(2.6, 0.25, f'{gap2}', va='center')

    ax.bar(x=4, height=0.5, width=0.5, color='yellow', alpha=0.7)
    ax.text(4.6, 0.25, f'{gap3}', va='center')

    ax.set_xlim(-0.5, 6)
    ax.set_ylim(-0.5, 1)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def runtime(s):
    hours, remainder = divmod(s, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Runtime: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))


def read_df_annot(rec_dir, filename):
    """
    Reads and processes the annotations from a JSON file.
    """
    # Mapping for beat symbols
    all_beats = {'N': 0, 'S': 1, 'V': 2, 'U': 3}
    
    # Extract the file extension
    file_extension = os.path.splitext(filename)[1]
    if file_extension.lower() == '.npy':
        filename = filename.split(".")[0]
    
    file_path = Path(rec_dir, filename + '.json')

    with open(file_path, 'r', encoding='UTF-8', errors='ignore') as f:
        data = json.loads(f.read())
    
    # Normalize JSON data into a DataFrame
    df = pd.json_normalize(data, record_path=['rpeaks'])
    
    # Create a new DataFrame for annotations
    df_annot = pd.DataFrame()
    df_annot['rpeak'] = df['sampleIndex'].astype(int)
    df_annot['annot'] = df['annotationValue'].map(all_beats)
    
    return df_annot


from matplotlib.ticker import MultipleLocator

def plot_signal(fileName, ecg_signal, fs, num_fragment, plot_signal_from_in_secs, plot_signal_to_in_secs,
                plot_save_dir=None, save_mark=None, recID=None,
                gap1_indices_secs=[], gap2_indices_secs=[], gap3_indices_secs=[],
                annot_df=None, flag_secs=True):

    # Convert seconds to sample indices
    plot_signal_from = int(plot_signal_from_in_secs * fs)
    plot_signal_to = int(plot_signal_to_in_secs * fs)

    # Extract signal portion
    signal_portion = ecg_signal[plot_signal_from:plot_signal_to]

    # Define x-axis
    if flag_secs:
        time_axis = np.arange(plot_signal_from, plot_signal_to) / fs
        x_label = "Time (s)"
    else:
        time_axis = np.arange(plot_signal_from, plot_signal_to)
        x_label = "Sample Index"

    x_start = time_axis[0]
    x_end = time_axis[-1]

    # Plotting
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, signal_portion, label=f'Fragment {num_fragment}')
    plt.xlabel(x_label)
    plt.ylabel("Amplitude")
    
    if recID is not None:
        plt.title(f"File: {fileName}  RecID: {recID}  Frag: {num_fragment}")
    else:
        plt.title(f"File: {fileName} Fragment: {num_fragment}")
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Annotated R-peaks
    if annot_df is not None and not annot_df.empty:
        fragment_df = annot_df[(annot_df['rpeak'] >= plot_signal_from) & (annot_df['rpeak'] < plot_signal_to)]
        rpeak_amps = ecg_signal[fragment_df['rpeak']]
        rpeak_x = fragment_df['rpeak'] / fs if flag_secs else fragment_df['rpeak']

        # Normal R-peaks
        normal_mask = fragment_df['annot'] == 0
        plt.plot(rpeak_x[normal_mask], rpeak_amps[normal_mask], "ro", label="_nolegend_", markersize=4)

        # Extrasystoles
        extras_mask = fragment_df['annot'] != 0
        if extras_mask.any():
            pred_to_mark = {1: "S", 2: "V", 3: "U", 4: "E"}
            sub_df = fragment_df[extras_mask]
            r_times = sub_df['rpeak'] / fs if flag_secs else sub_df['rpeak']
            r_amps = ecg_signal[sub_df['rpeak']]
            for x, y, p in zip(r_times, r_amps, sub_df['annot']):
                mark = pred_to_mark.get(p, "?")
                plt.text(x, y + 0.05, mark, fontsize=12, color='black', ha='center')
            plt.plot(r_times, r_amps, "ko", label="_nolegend_", markersize=6)

    # Helper to draw gaps
    def plot_gap_regions(gap_list, color):
        for pause_start_secs, pause_end_secs in gap_list:
            # Convert gap start and end to the same x-axis units
            if not flag_secs:
                pause_start_secs *= fs
                pause_end_secs *= fs
            if pause_start_secs < x_end and pause_end_secs > x_start:
                clip_start = max(pause_start_secs, x_start)
                clip_end = min(pause_end_secs, x_end)
                plt.axvspan(clip_start, clip_end, color=color, alpha=0.5)

    # Draw all 3 gap types
    plot_gap_regions(gap1_indices_secs, 'red')
    plot_gap_regions(gap2_indices_secs, 'blue')
    plot_gap_regions(gap3_indices_secs, 'yellow')

        # Extract the numerical part from the fileName
    base_name = os.path.basename(fileName)
    number_part = os.path.splitext(base_name)
    formatted_number_part = number_part[0] +  number_part[1].replace('.', '_') + '_' + save_mark 

# Papildomi tikai horizontalioje ašyje  
    
    # Set x-axis ticks to match your fragment start/end
    tick_spacing = (plot_signal_to - plot_signal_from) / 20  # or /10 for finer ticks
    if flag_secs:
        tick_spacing /= fs  # convert to seconds

    plt.gca().xaxis.set_major_locator(MultipleLocator(tick_spacing))    
    
# Add start and end labels without overwriting all ticks
    tick_positions = [plot_signal_from, plot_signal_to]
    if flag_secs:
        tick_positions = [x / fs for x in tick_positions]
    for tick in tick_positions:
        plt.axvline(tick, color='gray', linestyle='--', alpha=0.3)
        plt.text(tick, plt.ylim()[0], f"{tick:.1f}", va='bottom', ha='center', fontsize=8, color='gray')
                    
    
    # Saving plottings
    
    # Check if the directory is writable
    if plot_save_dir is not None:
        # Ensure the directory exists
        os.makedirs(plot_save_dir, exist_ok=True)
        
        # Check if the directory is writable
        if not os.access(plot_save_dir, os.W_OK):
            print(f"Error: The directory '{plot_save_dir}' is not writable.", file=sys.stderr)
        else:
            plot_filename = os.path.join(plot_save_dir, f"{num_fragment}_frag_{formatted_number_part}.png")
            plt.savefig(plot_filename)
    else:
        plt.show()

    plt.close()
           