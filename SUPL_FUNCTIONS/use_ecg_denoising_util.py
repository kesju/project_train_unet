import numpy as np
import argparse
import neurokit2 as nk
import json, os, h5py
from typing import Tuple, List

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


def convert_seconds_to_hms(total_seconds):
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return hours, minutes, seconds


def ecg_filter(ecg_signal, fp):
    """
    # signal_filter(signal, sampling_rate=1000, lowcut=None, highcut=None, method='butterworth', order=2, window_size='default', powerline=50, show=False)
    # Filter a signal using different methods such as “butterworth”, “fir”, “savgol” or “powerline” filters.
    # Apply a lowpass (if “highcut” frequency is provided), highpass (if “lowcut” frequency is provided)
    # or bandpass (if both are provided) filter to the signal.
   
    Parameters with example values:
    fp = {  'type': 'lowpass' or 'bandpass',
            'method':'butterworth',
            'order':5,
            'sampling_rate':200,
            'lowcut':0.5,
            'highcut':90 }
    """
    
    if fp['type'] == 'lowpass':
        # Apply lowpass filter to ecg signal.
        ecg_signal_flt = nk.signal_filter(signal=ecg_signal, sampling_rate=fp['sampling_rate'], lowcut=fp['lowcut'], method=fp['method'], order=fp['order'])
    else:
        # Apply bandpass filter to ecg signal.
        ecg_signal_flt = nk.signal_filter(signal=ecg_signal, sampling_rate=fp['sampling_rate'], lowcut=fp['lowcut'], method=fp['method'], order=fp['order'])
    return ecg_signal_flt





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


def find_ecg_outliers(signal, ekg_min, ekg_max, length_fragment):
   # Explanation:
# Fragment Creation:

# The function identifies the outlier values in the signal that are either above ekg_max or below ekg_min.
# For each outlier, it calculates the start and end indices of a fragment of length length_fragment centered
# around the outlier, unless the index is already part of an existing fragment.

# Efficiency Considerations:
# The function avoids searching for outliers in the indices that are already part of previously found fragments.
# This ensures that no outliers are counted twice, and no redundant fragments are created.

# Merging Fragments:
# After all fragments are found, the function checks for neighboring or overlapping fragments and merges
# them into single fragments.    
 
    fragments = []
    length_half = length_fragment // 2
    excluded_indices = set()

    def add_fragment(center_index):
        start = max(0, center_index - length_half)
        end = min(len(signal), center_index + length_half)
        fragments.append((start, end))
        excluded_indices.update(range(start, end))

    # Find outliers and construct fragments
    for i, value in enumerate(signal):
        if i in excluded_indices:
            continue

        if value > ekg_max or value < ekg_min:
            add_fragment(i)

    # Merge neighboring fragments
    merged_fragments = []
    for fragment in fragments:
        if not merged_fragments:
            merged_fragments.append(fragment)
        else:
            last_start, last_end = merged_fragments[-1]
            current_start, current_end = fragment
            if current_start <= last_end:  # Fragments overlap or are neighbors
                merged_fragments[-1] = (last_start, max(last_end, current_end))
            else:
                merged_fragments.append(fragment)

    return merged_fragments


def find_ecg_rdropouts(ecg_signal, r_peaks, len_frag, fs, HR_MIN, HR_MAX, step_size):
    """
    Analyze ECG signal using a sliding window approach, checking the number of R peaks in each window and
    returning fragments with abnormal peak counts.
    """
    # Ieškant intervalų su rspragomis, seka po intervalų su triukšminiais šuoliais ir didelėmis
    # osciliacijomis eliminavimo ir su surastais R pikų indeksais apdorojama slenkančiu langu.
    # Lango plotis nustatomas eksperimentiškai, jis turi nepersikloti su galimomis širdies tvinksnių pauzėmis,
    # pvz. 10 sek. (2000 EKG signalo reikšmių).
    # Langas slenkamas užduotu žingsniu (pvz. paslenkant kas 100 reikšmių), jame skaičiuojant R pikų skaičių ir lyginant
    # su ribiniais skaičiais, atitinkančiais širdies tvinksnių dažnumą HR max = 220 ir min = 20. Daroma prielaida,
    # kad jei surastų R pikų skaičius yra už ribų, jis yra iškreiptas dėl triukšmų signale arba nutrūkusio EKG signalo.
    # Fragmentai, kuriuose R pikų skaičius netelpa į tą diapazoną, yra pažymimi
    # kaip netinkami analizei. Pažymėti fragmentai, einantys vienas po kito, apjungiami į intervalus, kurie po to eliminuojami
    # iš sekos. 
    
    len_frag_sec = len_frag / fs
    min_r_peaks = HR_MIN * len_frag_sec / 60
    max_r_peaks = HR_MAX * len_frag_sec / 60

    # Initialize an empty list to store abnormal fragments
    abnormal_fragments = []
    
    # Initialize window and peak tracking variables
    start_idx = 0
    enter_idx = 0
    leave_idx = 0
    r_peaks_in_window = 0

    # Sliding window loop through the entire signal
    while start_idx + len_frag <= len(ecg_signal):
        end_idx = start_idx + len_frag

        # Update R peaks count in the window
        r_peaks_in_window, enter_idx, leave_idx = update_r_peaks_in_window(r_peaks_in_window, r_peaks, start_idx, end_idx, enter_idx, leave_idx)

        # Check if the number of R peaks is outside the normal range
        if r_peaks_in_window < min_r_peaks or r_peaks_in_window > max_r_peaks:
            abnormal_fragments.append((start_idx, end_idx))

        # Move the window forward by step_size
        start_idx += step_size

    # Handle the tail if the last fragment is smaller than the fragment length
    if start_idx < len(ecg_signal):
        end_idx = len(ecg_signal)
        len_tail_sec = (end_idx - start_idx) / fs
        min_r_peaks_tail = HR_MIN * len_tail_sec / 60
        max_r_peaks_tail = HR_MAX * len_tail_sec / 60

        # Update R peaks count in the tail window
        r_peaks_in_window, enter_idx, leave_idx = update_r_peaks_in_window(r_peaks_in_window, r_peaks, start_idx, end_idx, enter_idx, leave_idx)

        if r_peaks_in_window < min_r_peaks_tail or r_peaks_in_window > max_r_peaks_tail:
            abnormal_fragments.append((start_idx, end_idx))

    return combine_overlapping_fragments(abnormal_fragments)


def update_r_peaks_in_window(r_peaks_in_window, r_peaks, start_idx, end_idx, enter_idx, leave_idx):
    """
    Update the count of R peaks in the window by removing peaks that left and adding new peaks that entered.
    """
    # Remove peaks that have exited the window (to the left)
    while leave_idx < len(r_peaks) and r_peaks[leave_idx] < start_idx:
        r_peaks_in_window -= 1
        leave_idx += 1

    # Add peaks that have entered the window (from the right)
    while enter_idx < len(r_peaks) and r_peaks[enter_idx] < end_idx:
        r_peaks_in_window += 1
        enter_idx += 1
    
    return r_peaks_in_window, enter_idx, leave_idx


def combine_overlapping_fragments(abnormal_fragments):
    """
    Combine neighboring and overlapping abnormal fragments into one.
    """
    if not abnormal_fragments:
        return []
    
    combined_fragments = [abnormal_fragments[0]]

    for i in range(1, len(abnormal_fragments)):
        previous_fragment = combined_fragments[-1]
        current_fragment = abnormal_fragments[i]
        
        # Check if fragments are overlapping or neighboring
        if current_fragment[0] <= previous_fragment[1]:
            # Extend the previous fragment to include the current one
            combined_fragments[-1] = (previous_fragment[0], max(previous_fragment[1], current_fragment[1]))
        else:
            # Add the current fragment as a new one
            combined_fragments.append(current_fragment)
    
    return combined_fragments

def post_processing(signal_length, fs, distortion_indices, t_gap_max_secs=30, extra_interval_secs=30, t_start_gap_max_secs=10, t_end_gap_max_secs=10):

        if (len(distortion_indices) == 0):
            return []

        # Convert times in seconds to samples
        t_gap_max_samples = int(t_gap_max_secs * fs)
        extra_interval_samples = int(extra_interval_secs * fs)
        t_start_gap_max_samples = int(t_start_gap_max_secs * fs)
        t_end_gap_max_samples = int(t_end_gap_max_secs * fs)
        
        # Step 1: Merge intervals if the gap between them is less than t_gap_max_secs
        def merge_intervals(intervals, gap_max_samples):
            merged_intervals = []
            current_start, current_end = intervals[0]

            for next_start, next_end in intervals[1:]:
                if next_start - current_end <= gap_max_samples:
                    current_end = next_end  # Merge the intervals
                else:
                    merged_intervals.append((current_start, current_end))
                    current_start, current_end = next_start, next_end

            merged_intervals.append((current_start, current_end))
            return merged_intervals
        
        # Initial merge of intervals based on t_gap_max_secs
        distortion_indices.sort()  # Ensure intervals are sorted
        distortion_indices = merge_intervals(distortion_indices, t_gap_max_samples)

        # Step 2: Add extra ECG segments to start and end of each interval
        adjusted_intervals = []

        for start, end in distortion_indices:
            new_start = max(0, start - extra_interval_samples)  # Adjust start
            new_end = min(signal_length - 1, end + extra_interval_samples)  # Adjust end
            adjusted_intervals.append((new_start, new_end))
        
        # Step 3: Extend first interval to the start if it's closer than t_start_gap_max_secs
        if adjusted_intervals[0][0] <= t_start_gap_max_samples:
            adjusted_intervals[0] = (0, adjusted_intervals[0][1])

        # Step 4: Extend last interval to the end if it's closer than t_end_gap_max_secs
        if adjusted_intervals[-1][1] >= signal_length - t_end_gap_max_samples:
            adjusted_intervals[-1] = (adjusted_intervals[-1][0], signal_length - 1)

        # Step 5: Merge intervals again if needed
        final_intervals = merge_intervals(adjusted_intervals, t_gap_max_samples)

        return final_intervals

def adjust_noisy_intervals(signal: np.ndarray, intervals: List[Tuple[int, int]],
    main_window_size: int,  sliding_window_size: int, ) -> List[Tuple[int, int]]:
    """
    Adjust noisy intervals in a signal using a special noise adjustment function.
    Raises errors if index adjustments go beyond limits.

    Args:
        signal (np.ndarray): The signal array to process.
        intervals (List[Tuple[int, int]]): A list of tuples where each tuple contains the start and end indices of noisy intervals.
        main_window_size (int): The default window size for adjusting indices.
        sliding_window_size (int): A secondary window size used by the special noise adjustment function.
        minimum_main_window_size (int): The minimum allowable main_window_size to avoid boundary issues.
        adjust_function (Callable[[np.ndarray, int, int, int, str], int]): 
            A function that takes the signal, an index, main window size, sliding window size, 
            and an adjustment type ("start" or "end"), and returns the adjusted index.

    Returns:
        List[Tuple[int, int]]: The adjusted list of intervals with refined start and end indices.
    
    Raises:
        StartIndexOutOfBoundsError: If the adjusted start index goes below zero or overlaps with a previous interval.
        EndIndexOutOfBoundsError: If the adjusted end index exceeds the signal length or overlaps with the next interval.
    """
    
    # print("\n++++++++++++++++++++++++++++++++++++++++++++")
    # print(f"postprocessing.py: Adjusting parameters - sliding_window_size: {sliding_window_size}, main_window_size: {main_window_size}")
    # print(len(signal), type(signal), signal[0:10])
    # print(intervals)
    # print("++++++++++++++++++++++++++++++++++++++++++++\n")
        
    signal_length = len(signal)
    fs = 200.  # Sampling frequency in Hz
    
    for i, (start_idx, end_idx) in enumerate(intervals):
        # Try adjusting the start index (move to left)
        # print("\nis_valid_interval_for_left:", start_idx, end_idx)
        
        if is_valid_interval_for_left(start_idx, end_idx, main_window_size, sliding_window_size, signal_length):
            (new_start_idx_left, new_end_idx_left), max_correlation_value_left = adjust_noisy_interval_left(signal, (start_idx,end_idx), main_window_size, sliding_window_size)
            # print(f"new_start_idx_left, new_end_idx_left: ({new_start_idx_left}, {new_end_idx_left})")
            # print(f"new_start_idx_left_secs, new_end_idx_left_secs: ({new_start_idx_left/fs:.1f}, {new_end_idx_left/fs:.1f})")
            # print(f"max_correlation_value_left: {max_correlation_value_left}")
            
            # Ensure the new start index does not go below zero or overlap with the previous interval
            if new_start_idx_left < 0:
                raise StartIndexOutOfBoundsError(new_start_idx_left, 0)
            if i > 0 and new_start_idx_left < intervals[i - 1][1]:
                raise OverlappingIntervalError(new_start_idx_left, intervals[i - 1][1])
        else:
            intervals[i] = (start_idx, end_idx)
            # print(f"no shift - new_start_idx, new_end_idx: ({start_idx/fs:.1f}, {end_idx/fs:.1f})")
            continue

        if is_valid_interval_for_right(start_idx, end_idx, main_window_size, sliding_window_size, signal_length):

            # Try adjusting the end index (move to right)
            (new_start_idx_right, new_end_idx_right), max_correlation_value_right = adjust_noisy_interval_right(signal, (start_idx, end_idx), main_window_size, sliding_window_size)
            # print(f"new_start_idx_right, new_end_idx_right: ({new_start_idx_right/fs:.1f}, {new_end_idx_right/fs:.1f})")
            
            # Ensure the new end index does not exceed signal length or overlap with the next interval
            if new_end_idx_right >= signal_length:
                raise EndIndexOutOfBoundsError(new_end_idx_right, signal_length - 1)
            if i < len(intervals) - 1 and new_end_idx_right > intervals[i + 1][0]:
                raise OverlappingIntervalError(new_end_idx_right, intervals[i + 1][0])
        else:
            intervals[i] = (start_idx, end_idx)
            # print(f"no shift - new_start_idx, new_end_idx: ({start_idx/fs:.1f}, {end_idx/fs:.1f})")
            continue            
        
        if max_correlation_value_right > max_correlation_value_left:
            intervals[i] = (int(new_start_idx_right), int(new_end_idx_right))
            # print(f"\nshifted to right - new_start_idx, new_end_idx: ({new_start_idx_right/fs:.1f}, {new_end_idx_right/fs:.1f})")
        else:
            intervals[i] = (int(new_start_idx_left), int(new_end_idx_left))
            # print(f"\nshifted to left - new_start_idx, new_end_idx: ({new_start_idx_left/fs:.1f}, {new_end_idx_left/fs:.1f})")
        
    return intervals


def adjust_noisy_interval_left(signal: np.ndarray, interval: Tuple[int, int], main_window_size: int,
    sliding_window_size: int)-> Tuple[Tuple[int, int], float]:
 
    gab_start, gab_end = interval # Triukšmo intervalas
    # print(f"Adjusting left gab_start, gab_end: ({gab_start}, {gab_end})")
 
# Išskiriame signalo fragmentus pries triukšmo intervalą ir po triuksmo intervalo   

    # Raise an error if main_window_size is too big
    if gab_start - main_window_size <= 0:
        raise MainWindowSizeError(main_window_size)

    # Raise an error if sliding_window_size is too big
    if gab_end + sliding_window_size >= len(signal):
        raise SlidingWindowSizeError(sliding_window_size)

    # Form the main window before gab_start
    main_window = signal[gab_start - main_window_size:gab_start]
    # print(f"main_window ({len(main_window)}): {main_window[:10]}")

    # Form the sliding window after gab_end
    sliding_window = signal[gab_end:gab_end + sliding_window_size]
    # print(f"sliding_window ({len(sliding_window)}): {sliding_window[:10]}")

    #  Calculate cross-correlation using 'valid' mode
    cross_corr_valid = np.correlate(main_window, sliding_window, mode='valid')

    # Find the index of the maximum correlation value
    best_match_index_left = np.argmax(cross_corr_valid)
    max_correlation_value_left = cross_corr_valid[best_match_index_left]
    
    # print(f"best_match_index_left: {best_match_index_left}, max_correlation_value_left: {max_correlation_value_left}")
    
    best_match_index_right = main_window_size - best_match_index_left
    
    a1_idx1 = -best_match_index_right
    a1_idx2 = -best_match_index_right + len(sliding_window)
    if a1_idx2 == 0:
        a1_idx2 = -1
    a1_match = main_window[a1_idx1:a1_idx2]
    max_amplitude_index = np.argmax(a1_match)
    
    # print(f"best_match_index_right, max_amplitude_index: {(best_match_index_right, max_amplitude_index)}")
    
    gab_start =  gab_start - best_match_index_right + max_amplitude_index
    gab_end = gab_end + max_amplitude_index

    return (gab_start, gab_end), max_correlation_value_left    


def adjust_noisy_interval_right(signal: np.ndarray, interval: Tuple[int, int], main_window_size: int,
    sliding_window_size: int)-> Tuple[Tuple[int, int], float]:
 
    gab_start, gab_end = interval # Triukšmo intervalas
    
# Išskiriame signalo fragmentus pries triukšmo intervalą ir po triuksmo intervalo   

    # Raise an error if main_window_size is too big
    if gab_end + main_window_size > len(signal):
        raise MainWindowSizeError(main_window_size)

      # Raise an error if sliding_window_size is too big
    if gab_start - sliding_window_size <= 0:
        raise SlidingWindowSizeError(sliding_window_size)
    
    # Form the main window after gab_end
    main_window = signal[gab_end:gab_end + main_window_size]

    # Form the sliding window before gab_start 
    sliding_window = signal[gab_start - sliding_window_size:gab_start]

        #  Calculate cross-correlation using 'valid' mode
    cross_corr_valid = np.correlate(main_window, sliding_window, mode='valid')

    # Find the index of the maximum correlation value
    best_match_index_left = np.argmax(cross_corr_valid)
    max_correlation_value_right = cross_corr_valid[best_match_index_left]
    
    # best_match_index_right = main_window_size - best_match_index_left
    
    a2_match = main_window[best_match_index_left:best_match_index_left + len(sliding_window)]
    max_amplitude_index = np.argmax(a2_match)

    # print(f"best_match_index_left, max_amplitude_index: {(best_match_index_left, max_amplitude_index)}")

    gab_start =  gab_start - sliding_window_size + max_amplitude_index
    gab_end = gab_end + best_match_index_left + max_amplitude_index
   
    return (gab_start, gab_end), max_correlation_value_right    

def is_valid_interval_for_right(start_idx: int, end_idx: int, main_window_size: int, sliding_window_size: int, signal_length: int) -> bool:
    """
    Checks if the given start and end indices satisfy the conditions:
    - start_idx - sliding_window_size > 0
    - end_idx + main_window_size < signal_length
    
    Parameters:
    start_idx (int): Start index of the segment.
    end_idx (int): End index of the segment.
    sliding_window_size (int): Size of the sliding window.
    main_window_size (int): Size of the main window.
    signal_length (int): Total length of the signal.
    
    Returns:
    bool: True if the conditions are satisfied, otherwise False.
    """
    return (start_idx - sliding_window_size > 0) and (end_idx + main_window_size < signal_length)


def is_valid_interval_for_left(start_idx, end_idx, main_window_size, sliding_window_size, signal_length):
    """
    Checks if the interval (start_idx, end_idx) satisfies the given conditions:
    1. start_idx - main_window_size > 0
    2. end_idx + sliding_window_size < signal_length
    
    Parameters:
        start_idx (int): Start index of the interval
        end_idx (int): End index of the interval
        main_window_size (int): Size of the main window
        sliding_window_size (int): Size of the sliding window
        signal_length (int): Total length of the signal
    
    Returns:
        bool: True if the conditions are met, False otherwise
    """
    return (start_idx - main_window_size > 0) and (end_idx + sliding_window_size < signal_length)



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


def remove_episodes(signal, episode_indices):
    """
    Converts a signal by removing the intervals specified in episode_indices.

    Args:
        signal (np.array): The input signal to be processed.
        episode_indices (list of tuples): List of (start, end) indices representing gaps to be removed.

    Returns:
        np.array: The processed signal with pauses removed.
        list of tuples: The indices in the original signal that correspond to non-gap segments.
        list of tuples: The indices in the processed signal that correspond to non-gap segments.
    """
    output_signal = []
    input_no_episode_indices = []
    output_no_episode_indices = []

    current_index = 0
    output_index = 0
    for start, end in episode_indices:
        if current_index < start:
            segment_length = start - current_index
            output_signal.extend(signal[current_index:start])
            input_no_episode_indices.append((current_index, start - 1))
            output_no_episode_indices.append((output_index, output_index + segment_length - 1))
            output_index += segment_length
        current_index = end + 1

    if current_index < len(signal):
        segment_length = len(signal) - current_index
        output_signal.extend(signal[current_index:])
        input_no_episode_indices.append((current_index, len(signal) - 1))
        output_no_episode_indices.append((output_index, output_index + segment_length - 1))

    return np.array(output_signal), input_no_episode_indices, output_no_episode_indices



def find_outliers_rdropouts(ecg_signal_start):
   
    # ĮVAIRŪS PARAMETRAI SKAIČIAVIMUI
        
    # print("\n\nĮVAIRŪS PARAMETRAI SKAIČIAVIMUI")
    
    fs = 200  # Sampling frequency in Hz
    # print("fs:", fs)

    minimum_allowed_number_of_rpeaks = 80
    # print("minimum_allowed_number_of_rpeaks:", minimum_allowed_number_of_rpeaks)

    # Minimum allowed length of ECG signal for detecting of noises
    # Calculated for getting of 80 rpeaks for slow rhythm 40 bpm
    # 80 rpeaks = 80 beats = 80/40 min. = 2 min. = 120 secs
     
    minimum_allowed_length_of_ecg_signal_flt_secs = 120  # 120 secs signal
    # print("minimum_allowed_length_of_ecg_signal_flt_secs:", minimum_allowed_length_of_ecg_signal_flt_secs)
    minimum_allowed_length_of_ecg_signal_flt = minimum_allowed_length_of_ecg_signal_flt_secs * fs

    if (len(ecg_signal_start) < minimum_allowed_length_of_ecg_signal_flt):
        raise ValueError(f"Error: minimum allowed length of ecg_signal_start {minimum_allowed_length_of_ecg_signal_flt_secs} secs is not met.") 
   
    # Parametrai post processing funkcijai 
    t_gap_max_secs=10
    extra_interval_secs=10
    t_start_gap_max_secs=10
    t_end_gap_max_secs=10

            # Minkšto apjungimo parametrai: 

    # Nustatome pagrindinio lango main_window plotį, kuriame ieškosime apsijungimo vietos
    # ir šliaužiančio lango slicing_window plotį
    
    sliding_window_size_secs = 1.0 # slicing window
    sliding_window_size = int(sliding_window_size_secs*fs)

    main_window_size = sliding_window_size*5
    main_window_size_secs = main_window_size/fs

    # print(f"main_window_size: {main_window_size} ({main_window_size_secs:.1f} secs), sliding_window_size: {sliding_window_size} ({sliding_window_size_secs:.1f} secs)")
    # Notice:
    # sliding_window_size_secs must be less (at least 2 times) than main_window_size
    # main_window_size must be less than  t_gap_max_secs, extra_interval_secs, t_start_gap_max_secs, t_end_gap_max_secs

    # print("\nCheck if the conditions are met:")
    if 2*sliding_window_size_secs < main_window_size_secs:
        # print("OK. 2*sliding_window_size is less than main_window_size")
        flag_sliding_window_size_OK = True
    else:
        print("Not OK. 2*sliding_window_size is not less than main_window_size")
        flag_sliding_window_size_OK = False
    
    if all(main_window_size_secs < value for value in [t_gap_max_secs, extra_interval_secs, t_start_gap_max_secs, t_end_gap_max_secs]):
        # print("OK. main_window_size is less than all specified values: t_gap_max, extra_interval, t_start_gap_max, t_end_gap_max")
        flag_main_window_size_OK = True
    else:
        print("Not OK. main_window_size is not less than all specified values: t_gap_max, extra_interval, t_start_gap_max, t_end_gap_max")
        flag_main_window_size_OK = False

    if not (flag_main_window_size_OK and flag_sliding_window_size_OK):
        raise ValueError("Soft stitching parameters are not OK")

     # ČIA DĖSIME REZULTATUS

    results = {
      'status': {
          'success': True
      },
      'recording_id': 'test'
    }
    
    try:
            # IŠSKIRČIŲ (OUTLIERS) DETEKTAVIMAS

        ekg_min = -3
        ekg_max = 4
        length_fragment = 200

        outliers_indices = find_ecg_outliers(ecg_signal_start, ekg_min, ekg_max, length_fragment)

        if len(outliers_indices) > 0:

            # Koreguojame outliers_indices priderindami prie signalo ecg_signal_start ilgio
            outliers_indices = filter_indices_by_signal_length(len(ecg_signal_start), outliers_indices)
            # print(f"\nIšskirtys: {outliers_indices}")
            
            # Post processing of outliers indices
            outliers_indices = post_processing(len(ecg_signal_start), fs, outliers_indices, 
                            t_gap_max_secs, extra_interval_secs, t_start_gap_max_secs, t_end_gap_max_secs)
            # print(f"\nIšskirtys po apjungimo: {outliers_indices}")
            
            # Adjusting of outliers_indices to soft stitching
            outliers_indices_adjusted = adjust_noisy_intervals(ecg_signal_start, outliers_indices, main_window_size, sliding_window_size)
            # print(f"\nIšskirtys po korekcijos 'minkštam' sujungimui : {outliers_indices_adjusted}")

            # print("\nConvert ecg_signal_start to ecg_signal_no_outliers")
            ecg_signal, no_outliers_indices_orig, no_outliers_indices = remove_episodes(ecg_signal_start, outliers_indices_adjusted)
            len_ecg_signal_no_outliers = len(ecg_signal)
            
        else:
            # print(f"\nIšskirčių nerasta")
            ecg_signal = ecg_signal_start
            outliers_indices_adjusted = []
            no_outliers_indices = []
            no_outliers_indices_orig = []
            
        len_ecg_signal_no_outliers = len(ecg_signal)

                # RPIKŲ INDEKSŲ PAIEŠKA
                
        # Rpikų indeksų paieška iš sekos be išskirčių (ecg_signal_no_outliers)
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=200, correct_artifacts=False)
        rpeaks = rpeaks['ECG_R_Peaks']  # <class 'numpy.ndarray'>  array dtype: int64

                    # RSPRAGŲ (RDROPOUTS) PAIEŠKA IR PAŠALINIMAS

        HR_MIN = 20  # HR
        HR_MAX = 220 # HR
        len_frag = 2000  # Length of each fragment, 10 secs
        step_size = 100 

        rdropouts_indices = find_ecg_rdropouts(ecg_signal, rpeaks, len_frag, fs, HR_MIN, HR_MAX, step_size)

        if (len(rdropouts_indices) > 0):

            # Koreguojame rdropouts_indices priderindami prie signalo ecg_signal_no_oscillation ilgio
            rdropouts_indices = filter_indices_by_signal_length(len_ecg_signal_no_outliers, rdropouts_indices)

            # Post processing of rdropout indices
            rdropouts_indices = post_processing(len_ecg_signal_no_outliers, fs, rdropouts_indices, 
                            t_gap_max_secs, extra_interval_secs, t_start_gap_max_secs, t_end_gap_max_secs)

            # Adjusting of rdropouts_indices to soft stitching
            rdropouts_indices_adjusted = adjust_noisy_intervals(ecg_signal, rdropouts_indices, main_window_size, sliding_window_size)

            ecg_signal, no_rdropouts_indices_orig, no_rdropouts_indices = remove_episodes(ecg_signal, rdropouts_indices_adjusted)
            
            # Skaičiuojame rpeaks tiesiogiai iš signalo be rspragų
            _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=200, correct_artifacts=False)

        else:
            rdropouts_indices_adjusted = []
            rpeaks_no_rdropouts = rpeaks
            no_rdropouts_indices = []
            no_rdropouts_indices_orig = []

    # Klaidos skripto blokuose
    except ValueError as error:
        results['status']['success'] = False
        results['status']['error'] = str(error)
    
    # Unexpected error during script execution 
    except Exception as error:
        results['status']['success'] = False
        results['status']['error'] = f"Unexpected error: {str(error)}"
    
    # Rezultatai: gražinamas ecg_signal be išskirčių ir rspragų
    # print("\n\nIšskirčių ir rspragų detektavimas baigtas")
    
    return ecg_signal, outliers_indices_adjusted, rdropouts_indices_adjusted  

# Normalization function (matched to training script)
def normalize(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-8:
        raise ValueError("Signal has near-zero variance; returning zeros")
        return np.zeros_like(signal)
    return (signal - mean) / std

# Preprocessing function
def preprocess_ecg(ecg, segment_length, overlap, fs):
    
    # Apply bandpass filter - užkomentuota, nes filtras jau buvo pritaikytas visam signalui jį nuskaičius
    # ecg_filtered = bandpass_filter(ecg, low=0.5, high=40, fs=fs, order=2)
    
    # Normalize the signal
    ecg_normalized = normalize(ecg)
    # ecg_normalized = normalize(ecg_filtered)
    
    # Segment the ECG into overlapping windows
    step = int(segment_length * (1 - overlap))
    segments = []
    indices = []
    for start in range(0, len(ecg_normalized) - segment_length + 1, step):
        segment = ecg_normalized[start:start + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
            indices.append(start)
    segments = np.array(segments)
    
    # Reshape for U-Net input (shape: [batch, segment_length, 1])
    segments = segments[:, :, np.newaxis]
    return segments, indices, ecg_normalized

# Function to merge sticking or overlapping intervals
def merge_intervals(indices, segment_length):
    if not indices:
        return []
    # Convert segment start indices to intervals
    intervals = []
    for idx in sorted(indices):
        intervals.append([idx, idx + segment_length - 1])
    
    # Merge overlapping or adjacent intervals
    merged = []
    current = intervals[0]
    for next_interval in intervals[1:]:
        if next_interval[0] <= current[1] + 1:  # Overlapping or adjacent
            current[1] = max(current[1], next_interval[1])
        else:
            merged.append(current)
            current = next_interval
    merged.append(current)
    return merged

# Postprocessing function to detect and merge noisy fragments
def detect_noisy_fragments(original, denoised, indices, segment_length, overlap, threshold=0.1):
    noisy_indices = []
    residual = np.abs(original - denoised)
    
    # Identify noisy segments
    for i, start_idx in enumerate(indices):
        segment_residual = residual[start_idx:start_idx + segment_length]
        mean_residual = np.mean(segment_residual)
        if mean_residual > threshold:
            noisy_indices.append(start_idx)
       
    # Merge noisy intervals
    noisy_intervals = merge_intervals(noisy_indices, segment_length)
    return noisy_intervals

def merge_lists_of_tuples(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    # Sort intervals by start index
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current

        if curr_start <= prev_end + 1:  # Overlapping or adjacent
            merged[-1] = (prev_start, max(prev_end, curr_end))  # Merge
        else:
            merged.append(current)

    return merged

def run_ecg_denoising_pipeline(ecg_data, model, CONFIG, threshold):
    """
    Full ECG denoising pipeline using a trained model.

    Parameters:
    - ecg_data: np.ndarray, raw ECG signal
    - model: trained Keras model (e.g., U-Net)
    - CONFIG: dict, configuration parameters including:
        - FS: int, sampling frequency
        - SEGMENT_LENGTH: int, length of each segment for processing
        - OVERLAP: float, fraction of overlap between segments
    - threshold: float, noise detection threshold

    Returns:
    - denoised_signal: np.ndarray
    - noisy_intervals: list of (start, end) indices
    """
    
    fs = CONFIG['FS']
    segment_length = CONFIG['SEGMENT_LENGTH']
    overlap = CONFIG['OVERLAP']
    
    # Step 1: Preprocess
    segments, segment_indices, ecg_normalized = preprocess_ecg(ecg_data, segment_length,  overlap, fs)

    # Step 2: Predict
    denoised_segments = model.predict(segments, verbose=0)

    # Step 3: Reconstruct full signal
    denoised_signal = np.zeros_like(ecg_normalized)
    count = np.zeros_like(ecg_normalized)
    for i, start_idx in enumerate(segment_indices):
        end_idx = start_idx + segment_length
        denoised_signal[start_idx:end_idx] += denoised_segments[i, :, 0]
        count[start_idx:end_idx] += 1
    count[count == 0] = 1
    denoised_signal /= count

    # Step 4: Detect noisy intervals
    noisy_intervals = detect_noisy_fragments(ecg_normalized, denoised_signal, segment_indices, segment_length, overlap, threshold)
    noisy_indices = [tuple(interval) for interval in noisy_intervals]
    
    return denoised_signal, noisy_indices



# start0
#  +++++++++++++++++++++++++++++++++++++   TESTAVIMUI ++++++++++++++++++++++++++++++++++++++++++++++++

def test_outliers_rdropouts_detecting(dir, filename):
        
         # ĮVAIRŪS PARAMETRAI SKAIČIAVIMUI

    # print("\n\nTESTUOJAME IŠSKIRČIŲ (OUTLIERS) IR RSPRAGŲ (RDROPOUTS) DETEKTAVIMAS")
    
            # NUSKAITOME EKG SIGNALĄ
            
    fs = 200  # Sampling frequency in Hz
    filePath = os.path.join(dir, filename)        
    ecg_signal_orig = get_ecg_signal(filePath) 
    # print(f"\nFailas: {filePath}") 
    # ecg_signal_orig = np.array([])   # testas  
    if len(ecg_signal_orig) == 0:
        raise ValueError("The ECG signal is empty.")
    
    len_ecg_signal_orig_secs = len(ecg_signal_orig)/fs
    # print(f"\nlen(ecg_signal_orig): {len(ecg_signal_orig)} ({len_ecg_signal_orig_secs:.1f} secs)")
    hours, minutes, seconds = convert_seconds_to_hms(len_ecg_signal_orig_secs)
    # print(f"Hours: {hours:.1f}, Minutes: {minutes:.1f}, Seconds: {seconds:.1f}")


            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  FILTRUOJAME SIGNALĄ

    # Filtruojame ecg signalą
    # print(f"\nSignalas filtruotas")
    
    # ecg_signal_start = bandpass_filter(ecg_signal_orig)

    ecg_signal_start = nk.signal_filter(signal=ecg_signal_orig, sampling_rate=200, lowcut=0.5, highcut=None, method='butterworth', order=5)

    len_ecg_signal_flt_secs = len(ecg_signal_start)/fs
    # print(f"\nlen(ecg_signal_start): {len(ecg_signal_start)} ({len_ecg_signal_flt_secs:.1f} secs)")
    # print(ecg_signal_start[:10], ecg_signal_start[-10:])
    
    ecg_signal, outliers_indices_adjusted, rdropouts_indices_adjusted = find_outliers_rdropouts(ecg_signal_start)
    print("\nTesting of detecting of outliers and rdropouts starts...")
    print("filePath:", filePath)
    print(f"len(ecg_signal_start): {len(ecg_signal_start)} ({len(ecg_signal_start)/fs:.1f} secs)")
    print(f"Outliers indices (adjusted): {outliers_indices_adjusted}")
    print(f"R-Dropouts indices (adjusted): {rdropouts_indices_adjusted}")
    print(f"len(ecg_signal): {len(ecg_signal)} ({len(ecg_signal)/fs:.1f} secs)")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D U-Net Denoising Autoencoder for ECG")
    parser.add_argument('--dir', type=str, required=True, help='Path to directory with ECG .npy files')
    parser.add_argument('--filename', type=str, required=True, help='filename of testing ECG .npy file')
    args = parser.parse_args()
    
    test_outliers_rdropouts_detecting(args.dir, args.filename)
    # print("\n")
    # print(args.dir, args.filename)
    print("Testing completed.")
    
# python use_denoising_util.py --dir /home/kesju/DI/ZIVEO_2025/DUOMENYS_UPD/records_npy_all/ --filename 1001_4.npy
# python use_denoising_util.py --dir /home/kesju/DI/ZIVEO_2025/DUOMENYS_UPD/records_npy_all/ --filename 1031_18.npy