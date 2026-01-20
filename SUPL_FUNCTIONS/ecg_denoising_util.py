# Čia sudėtos visos funkcijos, kurios reikalingos triukšmų detektavimui ir vaizdavimui


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys
import h5py, json, math
from pathlib import Path
import numpy as np
import argparse
import json, os, h5py



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


# def plot_signal_write_plot_R_P_2(fileName, ecg_signal, fs, num_fragment, plot_signal_from_in_secs, plot_signal_to_in_secs, portion_length_in_secs,
#                                  plot_save_dir, save_mark,
#                                  gap1_indices_secs=[], gap2_indices_secs=[],
#                                  gap3_indices_secs=[], mark_indices_secs=[],
#                                  rpeak_indices_secs= [], ppeak_indices_secs= []):


def plot_signal_write_plot_R_P_1(fileName, ecg_signal, fs, num_fragment, plot_signal_from_in_secs, plot_signal_to_in_secs, portion_length_in_secs, 
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
           

# Define custom exceptions for index errors
class IndexAdjustmentError(Exception):
    pass

class StartIndexOutOfBoundsError(IndexAdjustmentError):
    def __init__(self, index: int, limit: int):
        super().__init__(f"Start index {index} is out of bounds (limit: {limit}).")

class EndIndexOutOfBoundsError(IndexAdjustmentError):
    def __init__(self, index: int, limit: int):
        super().__init__(f"End index {index} exceeds the signal length (limit: {limit}).")
        
class OverlappingIntervalError(IndexAdjustmentError):
    def __init__(self, index: int, neighbor_index: int):
        super().__init__(f"Adjusted index {index} overlaps with neighbor index {neighbor_index}.")

# Custom exception (optional, or you can use a built-in exception)
class MainWindowSizeError(ValueError):
    def __init__(self, main_window_size: int):
        super().__init__(f"Start index {main_window_size} is too big).")

# Custom exception (optional, or you can use a built-in exception)
class SlidingWindowSizeError(ValueError):
    def __init__(self, sliding_window_size: int):
        super().__init__(f"sliding_window_size {sliding_window_size} is too big).")



def parse_gaps(json_string):
    try:
        gaps = json.loads(json_string)
        # Ensure that each gap is a tuple
        return [tuple(gap) for gap in gaps]
    except (ValueError, TypeError) as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON format for gaps: {e}")


def plot_signal_write_plot_R_P_2(fileName, ecg_signal, fs, num_fragment, plot_signal_from_in_secs, plot_signal_to_in_secs, portion_length_in_secs,
                                            plot_save_dir, save_mark, gap1_indices_secs=[], gap2_indices_secs=[],
                                            gap3_indices_secs=[], mark_indices_secs=[], rpeak_indices_secs= [], ppeak_indices_secs= []):

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
    # portion_length_in_secs = math.ceil(portion_length/fs)
    # print(f"portion_length: {portion_length_in_secs} secs. ({portion_length})")


    # if gap_indices_secs:
    #     gap_indices = [(x * fs, y * fs) for x, y in gap_indices_secs] # indexes
    #     print(f"pause indices: {gap_indices}")
        
    # Assuming ecg_signal, ecg_length, fs, portion_length, plot_signal_from, and plot_signal_to are defined
    # t = np.arange(ecg_length)  # Time vector in seconds
    
    ecg_length = len(ecg_signal)
    t = np.arange(ecg_length) / fs  # Time vector in seconds

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
        
        plt.figure(figsize=(15, 5))
        plt.plot(t[start_idx:end_idx], ecg_signal[start_idx:end_idx], label=f"Portion {i+1}")
        plt.title(f"ECG Signal Fragment {num_fragment}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        
        
        # Highlight gap1 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap1_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='red', alpha=0.3)
        
        # Highlight gap2 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap2_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='blue', alpha=0.3)

        # Highlight gap3 intervals on the plot
        for (pause_start_secs, pause_end_secs) in gap3_indices_secs:
            if pause_start_secs < t[end_idx] and pause_end_secs > t[start_idx]:
                # Clip the outliers interval to the current plot limits
                clip_start = max(pause_start_secs, t[start_idx])
                clip_end = min(pause_end_secs, t[end_idx])
                # print(clip_start, clip_end)
                plt.axvspan(clip_start, clip_end, color='yellow', alpha=0.3)
  
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



def map_across_signals_forward(signal_index, tolerance=0, *mappings):
    """
    Maps an index across multiple signals using a combined economical segment mapping,
    allowing for a small tolerance at the edges.

    Args:
        signal_index (int): The index to be mapped.
        tolerance (int): The allowable deviation from the exact segment boundary.
        *mappings (list of tuples of lists): The segment boundary mappings to apply in sequence.

    Returns:
        int: The final mapped index in the target signal, or None if mapping fails.
    """
    
    for mapping in mappings:
        from_indices, to_indices = mapping
        found = False
        
        for (from_start, from_end), (to_start, to_end) in zip(from_indices, to_indices):
            # Apply tolerance to the boundaries
            if (from_start - tolerance) <= signal_index <= (from_end + tolerance):
                # If the index is within the tolerance margin, adjust it
                signal_index = to_start + max(0, min(signal_index - from_start, from_end - from_start))
                found = True
                break
        if not found:
            # raise ValueError(f"Error. Signal index {signal_index} not found in any segment.")
            return None  # If the index is not found in any segment, return None
    return signal_index

def map_indices_across_signals_reverse(indices, tolerance, *mappings):
    """
    Maps a list of index intervals across multiple signals using the provided mappings.
    
    Args:
        indices (list of tuples): List of tuples, each containing a start and end index (start, end).
        tolerance (int): Allowable deviation from the exact segment boundary.
        *mappings (list of tuples of lists): The segment boundary mappings to apply in sequence.
        
    Returns:
        list of tuples: The final mapped index intervals in the target signal, or None if any mapping fails.
    """
    
    # tolerance = 1
    mapped_indices = []
    for start, end in indices:
        mapped_start = map_across_signals_reverse_prt(start, tolerance, False, *mappings)
        mapped_end = map_across_signals_reverse_prt(end, tolerance, False, *mappings)
        
        if mapped_start is None or mapped_end is None:
            mapped_indices.append(None)  # If any mapping fails, append None
        else:
            mapped_indices.append((mapped_start, mapped_end))
    
    return mapped_indices

def map_indices_across_signals_forward(indices, tolerance, *mappings):
    """
    Maps a list of index intervals across multiple signals using the provided mappings.
    
    Args:
        indices (list of tuples): List of tuples, each containing a start and end index (start, end).
        tolerance (int): Allowable deviation from the exact segment boundary.
        *mappings (list of tuples of lists): The segment boundary mappings to apply in sequence.
        
    Returns:
        list of tuples: The final mapped index intervals in the target signal, or None if any mapping fails.
    """
    
    # tolerance = 1
    mapped_indices = []
    for start, end in indices:
        mapped_start = map_across_signals_forward(start, tolerance, *mappings)
        mapped_end = map_across_signals_forward(end, tolerance, *mappings)
        
        if mapped_start is None or mapped_end is None:
            mapped_indices.append(None)  # If any mapping fails, append None
        else:
            mapped_indices.append((mapped_start, mapped_end))
    
    return mapped_indices


def map_rpeaks_across_signals_reverse(r_peaks, tolerance, *mappings):
    
    r_peaks_orig = []
    for r_peak in r_peaks:
        r_peak_orig = map_across_signals_reverse_prt(r_peak, tolerance, False, *mappings)
        r_peaks_orig.append(r_peak_orig)
        # print(f"r_peak: {r_peak}  r_peak_orig: {r_peak_orig}")
    return np.array(r_peaks_orig)


def map_rpeaks_across_signals_forward(r_peaks, tolerance, *mappings):
    
    r_peaks_orig = []
    for r_peak in r_peaks:
        r_peak_orig = map_across_signals_forward(r_peak, tolerance, *mappings)
        r_peaks_orig.append(r_peak_orig)
        # print(f"r_peak: {r_peak}  r_peak_orig: {r_peak_orig}")
    r_peaks_orig = [rpeak for rpeak in r_peaks_orig if rpeak != None]    
    return np.array(r_peaks_orig)


def filter_mappings(*mappings):
    """
    Filters out mappings that are ([], []).

    Args:
        *mappings (list of tuples of lists): The segment boundary mappings to filter.

    Returns:
        list: A list of filtered mappings.
    """
    return tuple(mapping for mapping in mappings if mapping != ([], []))



def adapt_show_indices_for_mapped_signal_forward(indices, new_signal_length, new_width, tolerance, *filtered_mappings):
    """
    Adapts a list of index intervals based on signal mapping, ensuring new start indices are valid
    and clips the indices based on the new signal length.

    Args:
    indices (list of tuples): Original start and end indices.
    new_signal_length (int): The current length of the signal to clip indices.
    new_width (int): The amount to add to the start index.
    tolerance (float): Tolerance value for signal mapping.
    filtered_mappings (list): Mappings used to map indices across signals.

    Returns:
    list of tuples: Adapted and filtered list of index intervals.
    
    Raises:
    ValueError: If the mapped start index is None or if the filtered indices are invalid.
    """
    
    def get_mapped_start(start):
        """
        Maps the start index across signals and raises an error if the result is invalid.
        """
        mapped_start = map_across_signals_forward(start, tolerance, *filtered_mappings)
        if mapped_start is None:
            raise ValueError(f"Error: new_start is None for start index {start}")
        return mapped_start

    # Map and extend the end of each index
    adapted_indices = []
    for start, end in indices:
        new_start = get_mapped_start(start)
        adapted_indices.append((new_start, new_start + new_width))
        
    # Filter indices to ensure they are within the valid signal length
    filtered_indices = filter_indices_by_signal_length(new_signal_length, adapted_indices)
    
    # Raise error if filtered_indices is None
    if filtered_indices is None:
        raise ValueError("Error: All adapted indices exceed the signal length.")
    
    return filtered_indices




def get_distortion_indices_in_fragment(show_frag_start_idx, show_frag_end_idx, 
                                        outliers_indices, oscillation_indices, rdropout_indices):
    """
    Returns a list of tuples (start, end) for distortion intervals (outliers, oscillations, rdropouts)
    that overlap with the given fragment interval. If the start or end is outside the fragment, they will be set to None.
    
    Parameters:
    - show_frag_start_idx: Start index of the fragment
    - show_frag_end_idx: End index of the fragment
    - outliers_indices: List of tuples (start, end) for outlier intervals (using indexes)
    - oscillation_indices: List of tuples (start, end) for oscillation intervals (using indexes)
    - rdropout_indices: List of tuples (start, end) for rdropout intervals (using indexes)
    
    Returns:
    - List of tuples (start, end) for intervals inside the fragment. If only one bound exists, 
      the other is set to None.
    """
    
    # Function to process distortion intervals based on indices
    def filter_intervals(indices):
        result = []
        for start, end in indices:
            if show_frag_start_idx <= start <= show_frag_end_idx:
                # Start is within the fragment
                if end is None or end > show_frag_end_idx:
                    # End is outside the fragment, set to None
                    result.append((start, None))
                else:
                    # Both start and end are within the fragment
                    result.append((start, end))
            elif show_frag_start_idx <= end <= show_frag_end_idx:
                # End is within the fragment but start is outside
                result.append((None, end))
            elif start <= show_frag_start_idx and end >= show_frag_end_idx:
                # Interval fully overlaps the fragment
                result.append((None, None))
        return result

    # Process each distortion type
    outliers_intervals = filter_intervals(outliers_indices)
    oscillation_intervals = filter_intervals(oscillation_indices)
    rdropout_intervals = filter_intervals(rdropout_indices)

    # Combine all the intervals into one list
    all_distortion_intervals = outliers_intervals + oscillation_intervals + rdropout_intervals

    return all_distortion_intervals


# Helper function to check if an index belongs to a distortion interval
def is_in_distortion_intervals(index, distortion_intervals):
    """
    Check if a given index belongs to any distortion interval.
    
    Parameters:
    - index: The index to check
    - distortion_intervals: List of tuples (start, end) for distortion intervals
    
    Returns:
    - True if the index belongs to a distortion interval, otherwise False.
    """
    for start, end in distortion_intervals:
        if start is not None and end is not None and start <= index <= end:
            return True
        elif start is not None and end is None and index >= start:
            return True
        elif start is None and end is not None and index <= end:
            return True
    return False


def find_distortion_start_indices_in_fragment_forward(distortion_indices_flt_in_fragment, filtered_mappings_forward):
        
        distortion_starts_flt_in_fragment = [x for x, y in distortion_indices_flt_in_fragment if x is not None]
        print(f"distortion_starts_flt_in_fragment:  {distortion_starts_flt_in_fragment}")
        
        if (len(distortion_starts_flt_in_fragment)) > 0: 
            if len(filtered_mappings_forward) > 0:
                # Surandame triukšminių intervalų pradžių atitikmenis signale ecg_signal
                tolerance = 2
                distortion_starts_in_fragment = [map_across_signals_forward(distortion_start_flt_in_fragment, 
                                        tolerance, *filtered_mappings_forward) for distortion_start_flt_in_fragment in distortion_starts_flt_in_fragment]
            else:
                distortion_starts_in_fragment = distortion_starts_flt_in_fragment
            
        return distortion_starts_in_fragment
        
def make_mark_indices(distortion_start, mark_length, show_frag_start, show_frag_end):
    """
    Creates distortion indices centered at `distortion_start`, with a length of `2 * mark_length` total.
    Ensures that the resulting indices do not exceed the bounds defined by `show_frag_start` and `show_frag_end`.
    
    Parameters:
    - distortion_start: The center of the distortion interval
    - mark_length: The length to expand from the center (total interval length = 2 * mark_length)
    - show_frag_start: The lower bound of the fragment (start)
    - show_frag_end: The upper bound of the fragment (end)
    
    Returns:
    - A list containing a single tuple representing the bounded distortion interval (start, end)
    """

    # Calculate the proposed start and end of the distortion interval
    proposed_start = distortion_start - mark_length
    proposed_end = distortion_start + mark_length

    # Clip the start and end so that they do not go outside the fragment boundaries
    bounded_start = max(proposed_start, show_frag_start)  # Ensure start is not less than fragment start
    bounded_end = min(proposed_end, show_frag_end)  # Ensure end is not greater than fragment end

    # Return the bounded distortion interval as a list of tuples
    distortion_indices = (bounded_start, bounded_end)
    
    return distortion_indices

    

def find_fragments_with_distortion_indices(show_frag_indices, distortion_intervals, distortion_name):
    # Initialize the result list
    result = []

    # Loop through each fragment
    for i, (frag_start, frag_end) in enumerate(show_frag_indices, start=1):
        # Initialize lists to store the distortion start and end indices inside the current fragment
        distortion_starts_in_fragment = []
        distortion_ends_in_fragment = []

        # Check if any distortion interval starts or ends within the current fragment
        for distortion_start, distortion_end in distortion_intervals:
            # Check if the start of the distortion is inside the fragment
            if frag_start <= distortion_start < frag_end:
                distortion_starts_in_fragment.append(distortion_start)
            
            # Check if the end of the distortion is inside the fragment
            if frag_start <= distortion_end < frag_end:
                distortion_ends_in_fragment.append(distortion_end)

        # Only append if there are distortion starts in the fragment
        if distortion_starts_in_fragment:
            result.append((i, frag_start, frag_end, distortion_starts_in_fragment, distortion_ends_in_fragment, distortion_name))

    return result


def find_fragments_with_distortions(show_frag_indices, outliers_indices, oscillation_indices, rdropout_indices):
        result = {}

        # Helper function to determine if two intervals overlap
        def intervals_overlap(fragment, distortion):
            return fragment[0] <= distortion[1] and distortion[0] <= fragment[1]

        # Iterate over each fragment with its serial number (index)
        for idx, fragment in enumerate(show_frag_indices):
            # Initialize a set to accumulate distortion types for the current fragment
            distortions_found = set()

            # Check against each distortion list
            for distortion in outliers_indices:
                if intervals_overlap(fragment, distortion):
                    distortions_found.add('outliers')
            
            for distortion in oscillation_indices:
                if intervals_overlap(fragment, distortion):
                    distortions_found.add('oscillation')
            
            for distortion in rdropout_indices:
                if intervals_overlap(fragment, distortion):
                    distortions_found.add('rdropout')
            
            # If any distortions were found, add them to the result
            if distortions_found:
                result[idx] = tuple(distortions_found)

        # Convert the result dictionary to the required list of tuples format
        return [(idx+1, *distortions) for idx, distortions in result.items()]


# Function to remove tuples where the start is inside the previous tuple's interval
def remove_overlapping_tuples(intervals):
    filtered_intervals = []
    prev_start, prev_end = intervals[0]
    
    # Add the first interval to the filtered list
    filtered_intervals.append((prev_start, prev_end))
    
    for start, end in intervals[1:]:
        if start > prev_end:
            # Only add the tuple if the start is not inside the previous tuple's interval
            filtered_intervals.append((start, end))
            prev_start, prev_end = start, end
    
    return filtered_intervals


def create_show_fragments_with_distortions(ecg_signal_flt, outliers_indices_orig, oscillation_indices_orig, rdropout_indices_orig, portion_length):
    # Merge all distortion intervals
    distortion_intervals = outliers_indices_orig + oscillation_indices_orig + rdropout_indices_orig
    if (len(distortion_intervals) == 0):
        return None
    # Sort intervals first to ensure proper order
    distortion_intervals = sorted(distortion_intervals)

    # # Merge overlapping or contiguous intervals
    # merged_fragments = []
    # current_start, current_end = distortion_intervals[0]

    # for next_start, next_end in distortion_intervals[1:]:
    #     if next_start <= current_end:
    #         # If the next interval overlaps or touches the current one, extend the current interval
    #         current_end = max(current_end, next_end)
    #     else:
    #         # If they don't overlap, add the current interval and start a new one
    #         merged_fragments.append((current_start, current_end))
    #         current_start, current_end = next_start, next_end

    # # Add the final interval
    # merged_fragments.append((current_start, current_end))

    # print(f"\ndistortion_intervals: {distortion_intervals}")
    # print(f"merged_fragments: {merged_fragments}")

    # Initialize the array for storing fragments with distortions
    show_frag_indices_flt = []

    # Loop through each distortion interval
    for start, end in distortion_intervals:
        # Ensure the fragment starts as close to the distortion interval as possible
        fragment_start = max(0, start)
        fragment_end = min(len(ecg_signal_flt), fragment_start + portion_length)

        # Add the fragment start and end to the list
        show_frag_indices_flt.append((fragment_start, fragment_end))

    # Sort the intervals to ensure they are continually increasing
    show_frag_indices_flt = sorted(show_frag_indices_flt, key=lambda x: x[0])
    show_frag_indices_flt = remove_overlapping_tuples(show_frag_indices_flt)
    show_frag_indices_flt = filter_indices_by_signal_length(len(ecg_signal_flt), show_frag_indices_flt)
    
    # print(f"koreguotas show_frag_indices_flt: {show_frag_indices_flt}")
    print()

    return show_frag_indices_flt



def find_ecg_oscillations(ecg_signal, window_size, max_num_zero_crossings, min_value, max_value):
    """
    Find and merge high zero crossing fragments in an ECG signal, considering a value range for zero crossings.
    
    Parameters:
    - ecg_signal: The input ECG signal.
    - window_size: The size of each fragment to check for zero crossings.
    - max_num_zero_crossings: The threshold for considering a fragment as having "high" zero crossings.
    - min_value: The minimum threshold for the zero crossing range.
    - max_value: The maximum threshold for the zero crossing range.
    
    Returns:
    - A list of merged fragments with high zero crossings.
    """
    fragments = []

    # Initial fragment zero crossing count with range checking
    current_fragment = ecg_signal[:window_size]
    current_num_zero_crossings = count_zero_crossings(current_fragment, min_value, max_value)

    if current_num_zero_crossings > max_num_zero_crossings:
        fragments.append((0, window_size - 1))

    # Sliding window to update zero crossings count
    for i in range(1, len(ecg_signal) - window_size + 1):
        # Update zero crossings count by sliding the window
        if ((ecg_signal[i - 1] < min_value and ecg_signal[i] > max_value) or 
            (ecg_signal[i - 1] > max_value and ecg_signal[i] < min_value)):
            current_num_zero_crossings -= 1
        if ((ecg_signal[i + window_size - 2] < min_value and ecg_signal[i + window_size - 1] > max_value) or 
            (ecg_signal[i + window_size - 2] > max_value and ecg_signal[i + window_size - 1] < min_value)):
            current_num_zero_crossings += 1

        if current_num_zero_crossings > max_num_zero_crossings:
            fragments.append((i, i + window_size - 1))

    # Handle the last fragment if it's smaller than the window size
    remainder_start = len(ecg_signal) - window_size + 1
    if remainder_start < len(ecg_signal):
        remainder_fragment = ecg_signal[remainder_start:]
        remainder_num_zero_crossings = count_zero_crossings(remainder_fragment, min_value, max_value)
        if remainder_num_zero_crossings > max_num_zero_crossings:
            fragments.append((remainder_start, len(ecg_signal) - 1))

    # Step 2: Merge neighboring fragments
    merged_fragments = []
    if fragments:
        current_start, current_end = fragments[0]

        for start, end in fragments[1:]:
            if start <= current_end + 1:
                current_end = max(current_end, end)  # Extend the current fragment
            else:
                merged_fragments.append((current_start, current_end))
                current_start, current_end = start, end

        # Add the last fragment
        merged_fragments.append((current_start, current_end))

    return merged_fragments


def count_zero_crossings(signal, min_value, max_value):
    """
    Count the number of zero crossings in a signal, but only if the signal goes outside the specified range.
    
    Parameters:
    - signal: The signal fragment to check.
    - min_value: The minimum threshold for crossing.
    - max_value: The maximum threshold for crossing.
    
    Returns:
    - The number of zero crossings that occur outside the specified range.
    """
    # Convert the signal to a sign function, but only mark values outside the range as crossing
    signal_within_range = np.where((signal > min_value) & (signal < max_value), 0, np.sign(signal))
    zero_crossings = np.diff(signal_within_range)
    return np.sum(zero_crossings != 0)


def find_RR_intervals_and_print_characteristics(rpeaks, fs):
    # RR charakteristikos    

    # Find R index differences
    R_index_diff = np.diff(np.array(rpeaks))
    RR_intervals_secs = R_index_diff/fs

    # Find min and max values and their indexes
    min_value = np.min(RR_intervals_secs)
    min_index = np.argmin(RR_intervals_secs)
    max_value = np.max(RR_intervals_secs)
    max_index = np.argmax(RR_intervals_secs)
    
    # print(f"\nRR_intervals_secs")
    print(f"Min value: {min_value:.2f} secs  HR: {(60./min_value):.1f} bpm  Index: {min_index}")
    pri_1 = max(min_index - 5, 0)
    pri_2 = min(min_index + 5, len(RR_intervals_secs))
    for i, value in enumerate(rpeaks[pri_1:pri_2], start=pri_1):
        print(f"R_index: {i}  R_index_diff: {R_index_diff[i]}   HR: {(60.*fs/R_index_diff[i]):.1f}  EKG_index: {value}  Time: {value/fs:.2f} secs")
        
    print(f"\nMax value: {max_value:.2f} secs  HR: {(60./max_value):.1f} bpm  Index: {max_index}")
    pri_1 = max(max_index - 5, 0)
    pri_2 = min(max_index + 5, len(RR_intervals_secs))
    for i, value in enumerate(rpeaks[pri_1:pri_2], start=pri_1):
        print(f"R_index: {i}  R_index_diff: {R_index_diff[i]}   HR: {(60.*fs/R_index_diff[i]):.1f}  EKG_index: {value}  Time: {value/fs:.2f} secs")          




def detect_pauses(signal, min_pause_length, tol=1e-5):
    
    # raise ValueError("klaida funkcijoje: detect_pauses ") # testas
    
    pauses = []
    n = len(signal)
    i = 0
    
    while i < n:
        start = i
        while i < n - 1 and np.isclose(signal[i], signal[i + 1], atol=tol):
            i += 1
        end = i
        if (end - start + 1) >= min_pause_length:
            pauses.append((start, end))
        i += 1
    
    return pauses

def CheckJSONCompatibility(list): 
    # Check if the list is JSON-compatible by attempting to serialize it
    try:
        json_data = json.dumps(list)
        print("The list is JSON-compatible and can be serialized.")
        print("Serialized JSON data:", json_data)
    except (TypeError, ValueError) as e:
        print("The list is not JSON-compatible:", e)

# Define custom exceptions for index errors
class IndexAdjustmentError(Exception):
    pass

class StartIndexOutOfBoundsError(IndexAdjustmentError):
    def __init__(self, index: int, limit: int):
        super().__init__(f"Start index {index} is out of bounds (limit: {limit}).")

class EndIndexOutOfBoundsError(IndexAdjustmentError):
    def __init__(self, index: int, limit: int):
        super().__init__(f"End index {index} exceeds the signal length (limit: {limit}).")

class OverlappingIntervalError(IndexAdjustmentError):
    def __init__(self, index: int, neighbor_index: int):
        super().__init__(f"Adjusted index {index} overlaps with neighbor index {neighbor_index}.")

# Custom exception (optional, or you can use a built-in exception)
class MainWindowSizeError(ValueError):
    def __init__(self, main_window_size: int):
        super().__init__(f"Main_window_size {main_window_size} is too big).")

# Custom exception (optional, or you can use a built-in exception)
class SlidingWindowSizeError(ValueError):
    def __init__(self, sliding_window_size: int):
        super().__init__(f"sliding_window_size {sliding_window_size} is too big).")

def calculate_correlation_validate(a1, a2, window_size, shift):
    # Take the shifted window_size samples from a1 and the first window_size samples from a2
    main_window = a1[-shift:-(shift-window_size)] if shift != 0 else a1[-window_size:]
    sliding_window = a2[:window_size]
    
    # Ensure window_size doesn't exceed the length of a1 or a2
    window_size = min(window_size, len(a1), len(a2))
    
    # Compute the cross-correlation between the end of a1 and the beginning of a2
    correlation = np.correlate(main_window, sliding_window, mode='validate')
    
    return correlation 

def filter_indices_by_signal_length(signal_length, indices):
    """
    Filters out any ranges that exceed the current signal length, ensuring that indices are within
    the valid range (0 to signal_length - 1).
    
    Args:
    signal_length (int): The current length of the signal.
    indices (list of tuples): A list of tuples representing start and end indices of ranges.
    
    Returns:
    list of tuples: A filtered list of ranges that do not exceed the signal length.
    """
    max_valid_index = signal_length - 1  # Since signal indices are 0-based
    
    def clip_range(start, end):
        """
        Clips the range to ensure both start and end are within the valid index range.
        """
        if start > max_valid_index:
            return None  # Skip range if start exceeds signal length
        return (start, min(end, max_valid_index))  # Clip end if it exceeds
    
    # Filter and clip indices using the helper function
    filtered_indices = [clip_range(start, end) for start, end in indices if clip_range(start, end) is not None]
    
    # Return None if filtered indices are empty
    # return filtered_indices if filtered_indices else None
    return filtered_indices



def map_across_signals_reverse_prt(signal_index, tolerance=0, print_flag = False, *mappings):
    """
    Maps an index across multiple signals using a combined economical segment mapping,
    allowing for a small tolerance at the edges.

    Args:
        signal_index (int): The index to be mapped.
        tolerance (int): The allowable deviation from the exact segment boundary.
        *mappings (list of tuples of lists): The segment boundary mappings to apply in sequence.

    Returns:
        int: The final mapped index in the target signal, or None if mapping fails.
    """
    
    if print_flag:
        print(f"\nmappings: {mappings}")
        
    for mapping in mappings:
        from_indices, to_indices = mapping
        if print_flag:
            print(f"from_indices: {from_indices} to_indices: {to_indices}") 
        found = False
        for (from_start, from_end), (to_start, to_end) in zip(from_indices, to_indices):
            if print_flag:
                print(f"from_start: {from_start} from_end: {from_end}  to_start: {to_start} to_end: {to_end}")
            # Apply tolerance to the boundaries
            if (from_start - tolerance) <= signal_index <= (from_end + tolerance):
                # If the index is within the tolerance margin, adjust it
                signal_index = to_start + max(0, min(signal_index - from_start, from_end - from_start))
                found = True
                break
        if not found:
            raise ValueError(f"Error. Signal index {signal_index} not found in any segment.")
            # return None  # If the index is not found in any segment, return None
    return signal_index
    

def map_rpeaks_df_to_original(rpeaks_df_no_rdropout, rpeaks_orig):
    rpeaks_df_orig = rpeaks_df_no_rdropout.copy()
    if len(rpeaks_df_no_rdropout['rpeak']) == len(rpeaks_orig):
        # Change rpeak values in rpeaks_df with values from rpeaks_orig
        rpeaks_df_orig['rpeak'] = rpeaks_orig
    else:
        raise ValueError("Error: lengths of rpeaks_df and rpeaks_orig is not the same")   
    return rpeaks_df_orig


def map_peaks_to_original(peaks, tolerance, *mappings):
    peaks_orig = []
    
    for peak in peaks:
        if peak == -1:
            peaks_orig.append(-1)
            continue
    
        peak_orig = map_across_signals_reverse_prt(peak, tolerance, False, *mappings)
        
        # print(f"peak: {peak}  peak_orig: {peak_orig}")
        # print(f"mappings: {mappings}")
        
        peaks_orig.append(peak_orig)
        
    return peaks_orig

def shift_t_peaks(ECG_T_Peaks, shift, signal_length):
    shifted_peaks = []
    for peak in ECG_T_Peaks:
        if peak != -1:  # Check if the value is not -1
            # Apply the shift and ensure it does not exceed the signal length
            new_peak = min(peak + shift, signal_length - 1)  # Ensure within bounds
            shifted_peaks.append(new_peak)
        else:
            # Keep -1 values unchanged
            shifted_peaks.append(peak)
    return shifted_peaks

def map_delineation_to_original(delineation, tolerance, *mappings):
    """
    Maps the delineation peaks from processed an ECG signal to the original ECG signal.
    
    Parameters:
    delineation (dict): Dictionary containing lists of peaks from the processed ECG signal.
                        Expected keys: 'ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks'.
    *mappings (list of tuples of lists): The segment boundary mappings to apply in sequence.                        

    Returns:
    dict: Dictionary containing mapped peaks for the original ECG signal
    """
    delineation_orig = {}
    
    # List of expected keys in the delineation dictionary
    peak_keys = ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']

    # Iterate over each key and map the peaks
    for key in peak_keys:
        if key in delineation:
            delineation_orig[key] = map_peaks_to_original(delineation[key], tolerance, *mappings)
        else:
            raise KeyError(f"Key '{key}' not found in the delineation dictionary.")
    return delineation_orig


def map_paroxysmal_tachycardia_episodes_to_original(tachycardia_episodes, tolerance, *mappings):
    tachycardia_episodes_orig = []
    
    for episode in tachycardia_episodes:
        start_orig = map_across_signals_reverse_prt(episode["start"], tolerance, False, *mappings)
        end_orig = map_across_signals_reverse_prt(episode["end"], tolerance, False, *mappings)
        tachycardia_episodes_orig.append({"start": start_orig, "end": end_orig})
    
    return tachycardia_episodes_orig


def map_afib_episodes_to_original(afib_episodes, tolerance, *mappings):

    afib_episodes_orig = []
    
    for episode in afib_episodes:
        start_orig = map_across_signals_reverse_prt(episode["start"], tolerance, False, *mappings)
        end_orig = map_across_signals_reverse_prt(episode["end"], tolerance, False, *mappings)
        afib_episodes_orig.append({"start": start_orig, "end": end_orig})
    
    return afib_episodes_orig



def map_pause_episodes_to_original(pauses, tolerance, *mappings):
    
    pauses_orig = []
    
    for episode in pauses:
        onset_orig = map_across_signals_reverse_prt(episode["onset_sampleno"], tolerance, False, *mappings)
        episode_orig = episode.copy()  # Copy the episode to retain other attributes
        episode_orig["onset_sampleno"] = onset_orig
        pauses_orig.append(episode_orig)
    
    return pauses_orig


def update_indices(mark_indices, removed_intervals, original_length, tolerance=1):
    """
    Adjusts the indices of marks after removal of noise episodes, considering signal length and tolerance.

    Parameters:
        mark_indices (list of int): Indices of marks before noise removal.
        removed_intervals (list of tuples): List of (start, end) intervals that were removed.
        original_length (int): Length of the signal before noise removal.
        tolerance (int): Maximum allowable shift for a mark to be kept.

    Returns:
        tuple: (Updated mark indices within valid range, New signal length after removal)
    """
    updated_marks = []
    total_removed = sum(e - s + 1 for s, e in removed_intervals)
    new_length = original_length - total_removed  # Update signal length after removal
    
    for mark in mark_indices:
        if mark >= original_length:  # Ignore marks that were already out of bounds
            continue 
        
        # print(f"\nmark: {mark}")
        
        shift = 0
        closest_valid_index = None

        for start, end in removed_intervals:
            # print(f"mark: {mark}  start: {start}, end: {end}")
            if mark > end:
                shift += (end - start + 1)  # Track removed samples before the mark
                # print(f"shift: {shift}")
            elif start <= mark <= end:  # Mark falls inside a removed interval
                before = start - 1  # Nearest valid index before the interval
                after = end + 1  # Nearest valid index after the interval
                
                # print(f"before: {before}, after: {after}")
                
                # Check if shifting within tolerance is possible
                if 0 <= before and abs(mark - before) <= tolerance:
                    closest_valid_index = before
                elif after < original_length and abs(mark - after) <= tolerance:
                    closest_valid_index = after
                
                # print(f"closest_valid_index: {closest_valid_index}")
                
                # If neither option is within tolerance, discard the mark
                if closest_valid_index is None:
                    mark = None
                else:
                    mark = closest_valid_index
                
                break

        if mark is not None:
            new_mark = mark - shift
            if 0 <= new_mark < new_length:  
                updated_marks.append(new_mark)  # Keep only valid indices

    return updated_marks, new_length


def get_mark_indices(marks, width, signal_length):
    """
    Generates a list of index intervals [(idx-width, idx+width)] for each mark,
    adjusting for boundary conditions. Ignores marks that are out of bounds.

    Parameters:
        marks (list of int): List of mark indices.
        width (int): Half-width for each interval.
        signal_length (int): Length of the signal.

    Returns:
        list of tuples: List of adjusted intervals.
    """
    indices = []
    
    for idx in marks:
        if idx < 0 or idx >= signal_length:
            continue  # Ignore invalid marks

        start = max(0, idx - width)  # Ensure start is not below 0
        end = min(signal_length - 1, idx + width)  # Ensure end is within signal length
        indices.append((start, end))
    
    return indices


