
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
    
    # len_ecg_signal_orig_secs = len(ecg_signal_orig)/fs
    # print(f"\nlen(ecg_signal_orig): {len(ecg_signal_orig)} ({len_ecg_signal_orig_secs:.1f} secs)")
    # hours, minutes, seconds = convert_seconds_to_hms(len_ecg_signal_orig_secs)
    # print(f"Hours: {hours:.1f}, Minutes: {minutes:.1f}, Seconds: {seconds:.1f}")


            # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++  FILTRUOJAME SIGNALĄ

    # Filtruojame ecg signalą
    # print(f"\nSignalas filtruotas")
    
    ecg_signal_start = bandpass_filter(ecg_signal_orig)
    # ecg_signal_start = nk.signal_filter(signal=ecg_signal_orig, sampling_rate=200, lowcut=fp['lowcut'], method=fp['method'], order=fp['order'])
    
    # len_ecg_signal_flt_secs = len(ecg_signal_start)/fs
    # print(f"\nlen(ecg_signal_start): {len(ecg_signal_start)} ({len_ecg_signal_flt_secs:.1f} secs)")
    # print(ecg_signal_start[:10], ecg_signal_start[-10:])
    
    find_outliers_rdropouts(ecg_signal_start)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a 1D U-Net Denoising Autoencoder for ECG")
    parser.add_argument('--dir', type=str, required=True, help='Path to directory with ECG .npy files')
    parser.add_argument('--filename', type=str, required=True, help='filename of testing ECG .npy file')
    args = parser.parse_args()
    
    test_outliers_rdropouts_detecting(args.dir, args.filename)
    
# python use_denoising_util.py --dir /home/kesju/DI/ZIVEO_2025/DUOMENYS_UPD/records_npy_all/ --filename 1001_4.npy
# python use_denoising_util.py --dir /home/kesju/DI/ZIVEO_2025/DUOMENYS_UPD/records_npy_all/ --filename 1031_18.npy       
