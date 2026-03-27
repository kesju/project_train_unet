from typing import Any, Dict
from pathlib import Path
import argparse

# naudosime funkcijas iš zive_data_read_utils.py
from zive_data_read_utils import load_ecg_npy, list_ecg_records, read_json_file


try:
    # <-- adjust this import to match your project structure if needed
    from ecg_denoising_pipeline import (
        ECGDenoisingPipeline,
        load_denoising_config_yaml,
        DenoisingPipelineConfig,
        check_denoising_config,
        resolve_model_path
    )
except Exception as exc:
    raise ImportError(
        "Cannot import run_denoising_pipeline. Update the import to match your project.\n"
        f"Original error: {exc}"
    ) from exc

# If you have a config checker, import it; otherwise we just keep the cfg path.
try:
    from ecg_denoising_pipeline.steps import check_denoising_config
except Exception:
    check_denoising_config = None

from record_noise_stats import calc_noise_stats_from_denoised_result

try:
    from ecg_denoising_pipeline.steps import check_denoising_config
except Exception:
    check_denoising_config = None

try:
    from ecg_ectopy_pipeline import (
        ECGEctopyPipeline,
        EctopyPipelineConfig,
        EctopyPipelineResult,
        load_ectopy_config_yaml,
    )
except Exception as exc:
    raise ImportError(
        "Cannot import ECG ectopy pipeline helpers. "
        "Update imports to match your project structure.\n"
        f"Original error: {exc}"
    ) from exc


def prepare_denoising_pipeline(
    denoising_config_path: Path,
    denoising_model_dir: Path,
    disable_motions: bool = False,
) -> DenoisingPipelineConfig:
    if not denoising_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {denoising_config_path}")
    if not denoising_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {denoising_model_dir}")

    cfg = load_denoising_config_yaml(str(denoising_config_path))
    if check_denoising_config is not None:
        check_denoising_config(cfg)

    cfg.motions.model_name = resolve_model_path(denoising_model_dir, cfg.motions.model_name)
    cfg.motions.enabled = not disable_motions
    return cfg


def create_unet_marker(cfg_denoising: DenoisingPipelineConfig) -> str:
    """Create a marker string based on the UNet model name and threshold for output column naming.""" 
    unet_model_name = Path(cfg_denoising.motions.model_name).name
    # print("\nunet_model_name:", unet_model_name, type(unet_model_name))
    MARKER = unet_model_name.removeprefix("resunet_ecg").removesuffix(".keras")  # -> "_1024_0_5_3_7"
    threshold = cfg_denoising.motions.threshold
    threshold_str = str( threshold).replace('.', '_')
    MARKER += f"_{threshold_str}"
    return MARKER


def prepare_ectopy_pipeline(
    ectopy_config_path: Path,
    ectopy_model_dir: Path,
) -> EctopyPipelineConfig:
    
    """Entry point when the ectopy pipeline is used ad-hoc (e.g. in notebooks)."""
    
    cfg: EctopyPipelineConfig = load_ectopy_config_yaml(ectopy_config_path)

    cfg.ectopy.model_name = resolve_model_path(ectopy_model_dir, cfg.ectopy.model_name)
    cfg.ectopy.scaler_name = resolve_model_path(ectopy_model_dir, cfg.ectopy.scaler_name)

    cfg.ectopy.ectopy_removing = False

    # print("[ECTOPY] Model:", cfg.ectopy.model_name)
    # print("[ECTOPY] Scaler:", cfg.ectopy.scaler_name)
    # print("[ECTOPY] Ectopy removing enabled:", bool(cfg.ectopy.ectopy_removing))

    return cfg

def compute_ectopy_stats(res_ectopy: EctopyPipelineResult | None) -> Dict[str, int]:
    """
    Count ectopic beat classes from rpeaks_on_denoised_df.

    Expected mapping in column 'pred':
        0 -> ectN
        1 -> ectS
        2 -> ectV
        3 -> ectU
        other values are ignored
    """
    stats = {"ectN": 0, "ectS": 0, "ectV": 0, "ectU": 0}

    if res_ectopy is None:
        return stats

    rpeaks_on_denoised_df = res_ectopy.rpeaks_on_denoised_df
    if rpeaks_on_denoised_df is None:
        return stats

    if "pred" not in rpeaks_on_denoised_df.columns:
        return stats

    counts = rpeaks_on_denoised_df["pred"].value_counts()

    return {
        "ectN": int(counts.get(0, 0)),
        "ectS": int(counts.get(1, 0)),
        "ectV": int(counts.get(2, 0)),
        "ectU": int(counts.get(3, 0)),
    }

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate accuracy of ectopic beat detection on ZIVE data")
    ap.add_argument("--dir", type=Path, required=True, help="Directory containing ECG .npy files and corresponding JSON metadata")    
    ap.add_argument("--exclude-list", type=Path, help="File containing list of ECG basenames to exclude (one basename per line, without extensions)")    
    ap.add_argument("--cfg-denoising", type=Path, required=True, help="Denoising config path")
    ap.add_argument("--unet-model-dir", type=Path, required=True, help="Model directory for denoising/motions")
    ap.add_argument("--disable-motions", action="store_true", help="Disable motions stage in denoising pipeline")
    ap.add_argument("--cfg-ectopy", type=Path, required=True, help="Ectopy config path")
    ap.add_argument("--ectopy-model-dir", type=Path, required=True, help="Model directory for ectopy detection")
    ap.add_argument("--fs", type=int, default=200, help="Sampling frequency (Hz). Default: 200")
    ap.add_argument("--quiet", action="store_true", help="Silence stdout during denoising/stats")
    ap.add_argument("--denoising", action="store_true", help="Silence stdout during denoising/stats")

    args = ap.parse_args()

    src = args.dir
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"--dir must be an existing directory. Got: {src}")
    

    #               ++++++++++++++++++++++++ ANALIZUOJAMŲ ĮRAŠŲ SĄRAŠAS

    scan_result = list_ecg_records(
        folder=src,
        data_format="auto",
        exclude_list=args.exclude_list,
    )

    records = scan_result.records
    summary = scan_result.summary

    print(f"total_json     : {summary.total_json}")
    print(f"excluded       : {summary.excluded}")
    print(f"matched        : {summary.matched}")
    print(f"unmatched_json : {summary.unmatched_json}")
    print(f"records returned: {len(records)}")

    # Naudosime šiuos signalus ir metadata toliau, kad įvertintume ektoopinių dūžių aptikimo tikslumą.
    # Galime naudoti metadata, kad sužinotume, kur yra ektoopiniai dūžiai, o signalą - kad patikrintume,
    # ar mūsų aptikimo algoritmas juos teisingai identifikuoja. 

        
    #               ++++++++++++++++++++++++ PIPELINE PARENGIMAS

    if (not args.denoising):
            print("\nDenoising is disabled. Existing Excel values in out/rdr/noi/tp_pct and ectN/ectS/ectV/ectU will be kept unchanged.")
            MARKER = ""
            noise_stats = {}
            denoising_model_dir = "Not used"
            cfg_denoising = "Not used"
            cfg_ectopy = "Not used"
            ectopy_pipe = None
            denoising_pipe = None
            denoising_pipe_disabled_motions = None
    else:
        print("\nDenoising will be performed for each record using the specified config and model.")

        # prepare the denoising pipeline (if needed) ++++++++++++++++++++++++++++++++++++++
        
        #  Preparing the Denoising pipeline with configuration and model paths
        print("\n*****Denoising pipeline config:")

        cfg_denoising_path = args.cfg_denoising
        denoising_model_dir = args.unet_model_dir
        print(f"\nDenoising config path: {cfg_denoising_path}")
        print(f"Denoising model directory: {denoising_model_dir}")
        
        cfg_denoising = prepare_denoising_pipeline(cfg_denoising_path, denoising_model_dir)
        MARKER = create_unet_marker(cfg_denoising)
        print(f"\nUNet marker: {MARKER}\n")
        denoising_pipe = ECGDenoisingPipeline(cfg_denoising)
        
        cfg_denoising_disabled_motions = prepare_denoising_pipeline(
        denoising_config_path=cfg_denoising_path,
        denoising_model_dir=denoising_model_dir,
        disable_motions=True,
        # disable_motions=args.disable_motions,
        )
        denoising_pipe_disabled_motions = ECGDenoisingPipeline(cfg_denoising_disabled_motions)
        
        # PREPARE ECTOPY DETECTION PART 
        
        cfg_ectopy = prepare_ectopy_pipeline(
            ectopy_config_path=args.cfg_ectopy,
            ectopy_model_dir=args.ectopy_model_dir
        )
        ectopy_pipe = ECGEctopyPipeline(cfg_ectopy)
        
        
    #               ++++++++++++++++++++++++ TRIUKŠMŲ VALYMAS IR EKTOPINIŲ DŪŽIŲ DETEKTAVIMAS

    print(f"Found {len(records)} matched records")
    for rec in records[:5]:
        print(rec.basename, rec.ecg_path.name, rec.json_path.name)
        
        if rec.ecg_path is None:
            msg = f"No matching ECG file for JSON '{rec.json_path.name}'"
            continue

        metadata: Dict[str, Any] = {}
        metadata = read_json_file(rec.json_path)
        
        # signal = load_ecg_npy(rec.ecg_path)
        # n_samples = int(signal.shape[0])
        # print(rec.basename, signal.shape)
        
        
        if args.denoising and denoising_pipe is not None:
            try:
                # With activated motions stage:
                x = load_ecg_npy(rec.ecg_path)

                # denoising and getting noise stats
                res_denoising = denoising_pipe.run(x, gaps_indices=[])
                noise_stats = calc_noise_stats_from_denoised_result(res_denoising) or {}
                print(
                    f"Noise stats for {rec.ecg_path.name}: "
                    f"out={noise_stats['out']}, rdr={noise_stats['rdr']}, "
                    f"noi={noise_stats['noi']}, tp_pct={noise_stats['tp_pct']:.1f} "
                    f"with enabled detecting motions"
                )
                print()

                # Without detecting motions stage (for comparison/debugging):
                x = load_ecg_npy(rec.ecg_path)

                res_denoising_disabled_motions = denoising_pipe_disabled_motions.run(x, gaps_indices=[])
                noise_stats_disabled_motions = calc_noise_stats_from_denoised_result(
                    res_denoising_disabled_motions
                ) or {}
                print(
                    f"Noise stats for {rec.ecg_path.name}: "
                    f"out={noise_stats_disabled_motions['out']}, "
                    f"rdr={noise_stats_disabled_motions['rdr']}, "
                    f"noi={noise_stats_disabled_motions['noi']}, "
                    f"tp_pct={noise_stats_disabled_motions['tp_pct']:.1f} "
                    f"with disabled detecting motions"
                )

                # detecting ectopies and getting ectopy stats
                res_ectopy = ectopy_pipe.run(res_denoising_disabled_motions, fs=args.fs)
                ectopy_stats = compute_ectopy_stats(res_ectopy)
                print(f"ectopy_stats={ectopy_stats} with disabled detecting motions")

            except Exception as exc:
                print(f"WARN: failed denoising/stat extraction for {rec.ecg_path.name}: {exc}")
                noise_stats = {}
                ectopy_stats = None


    # Dabar turėtume turėti noise_stats ir ectopy_stats kiekvienam įrašui, kuriuos galime palyginti su metadata,
    # kad įvertintume ektoopinių dūžių aptikimo tikslumą. Galime patikrinti, ar aptikti ektoopiniai dūžiai sutampa su tais, kurie nurodyti metadata, ir apskaičiuoti tikslumą, jautrumą, specifiškumą ir pan.    

    
    
"""
    # Įvertiname ectopijų aptikimo tikslumą naudojant annotacijas ir predikcijas

    # gabalas iš /home/kesju/DI/2025_ZIVEO/S-ITP-25-9/PROJECT_ECG_ECTOPY_PIPELINE/cli_ectopy.ipynb

    # Klaidos skaičiavimas
    # dalis paimta iš TEST_VU_CNN/zive_aritmijos_klasifikacija_NN_algoritmas_one_keras3.ipynb

    from ecg_denoising_pipeline import read_df_annot
    from ecg_ectopy_pipeline import (
        merge_rpeaks_with_annotations,
        print_classification_results,
        evaluate_binary_classification,
    )

    # from step_beats_no_ectopy import map_rpeaks_denoised_to_start


    # +++++++  RPIKŲ IR EKSTRASYSTOLIŲ INDEKSŲ PERSKAIČIAVIMAS Į PRADINĮ SIGNALĄ +++++++++++++++++++

    # rpikų indeksų su klasių numeriais perskaičiavimas į pradinių duomenų signalą ecg_start

    # rezultatas: rpeaks_on_start_df

    # pirmiausiai perskaičiuojame rpeaks_on_denoised į rpeaks_on_start
    # naudojame žemėlapius iš denoising pipeline: res.map_motions, res.map_rdropouts, res.map_outliers, res.map_gaps

    # nuskaitome anotoutacijas iš medikų  ir sulyginame su ML klasifikacija

    # 1. Iš išvalyto nuo triukšmų signalo randame R-peak'us rpeaks_on_denoised
    #  ir remapiname į R-peak'us į pradinį signalą
    rpeaks_on_denoised = res_ectopy.rpeaks_on_denoised_df["rpeak"].astype(int).tolist()
    rpeaks_on_start, rpeaks_on_gap = map_mark_denoised_to_start( rpeaks_on_denoised, res_denoising )
    rpeaks_on_start = np.array(rpeaks_on_start, dtype=int)

    # Build dataframe in 'start' coordinates (keeps Index and pred)
    rpeaks_on_start_df = res_ectopy.rpeaks_on_denoised_df.copy()
    rpeaks_on_start_df["rpeak"] = rpeaks_on_start
    # Backward-compatible alias if used later in the notebook
    rpeaks_df_start = rpeaks_on_start_df

    print(f"Mapped {len(rpeaks_on_start_df)} R-peaks from final -> start")
    # print("\nrpeaks_on_start_df:")
    # print(rpeaks_on_start_df.head(10))


                # ++++++++++++++++++++++++++  ECG PŪPSNIŲ KLASIFIKACIJOS SULYGINIMAS SU MEDIKŲ ANOTACIJOMIS 
    
    # Nuskaitome paciento įrašo medikų anotacijas atr_symbol_orig ('N', 'S', 'V', 'U')
    # ir jų indeksus atr_sample_orig (rpeaks vietas signal masyve)
    print(f"\nfileName: {file_name}")
    annot_df = read_df_annot(data_dir, file_name)
    print(f"\nlen(annot_df): {len(annot_df)}")
    # print("\nannot_df:")
    # print(annot_df.head(20))

    tolerance = 20
    print("\ntolerance:", tolerance)

        # Merge the dataframes
    df_matched, df_unmatched, not_found_count, description = merge_rpeaks_with_annotations(rpeaks_on_start_df,
                                                                        annot_df, tolerance=tolerance)

    print(description)
    print(f"\nlen(df_matched): {len(df_matched)}")
    df_matched = df_matched.sort_values(by='diff', ascending=False)
    # print(df_matched.head(10))
    print(f"\nlen(df_unmatched): {len(df_unmatched)}")
    df_unmatched["rpeak_sec"] = df_unmatched["rpeak"] / fs
    df_unmatched["closest_rpeak_annot_sec"] = df_unmatched["closest_rpeak_annot"] / fs
    # print(df_unmatched.head(20))

    # Remove annot == 3 from df_matched
    removed_count = (df_matched['annot'] == 3).sum()
    df_matched = df_matched[df_matched['annot'] != 3].reset_index(drop=True)
    print(f"\nRemoved {removed_count} rows with annot == 3")

    # (unique_labels_annot, counts_annot) = np.unique(df_matched['annot'].values, return_counts=True)
    counts = df_matched['annot'].value_counts(dropna=False)
    unique_labels_annot = counts.index.to_numpy()
    counts_annot = counts.to_numpy()
    print("\nLabels for annot: ", unique_labels_annot, counts_annot, "Total:", counts_annot.sum())

    # (unique_labels_pred, counts_pred) = np.unique(df_matched['pred'].values, return_counts=True)
    counts = df_matched['pred'].value_counts(dropna=False)
    unique_labels_pred = counts.index.to_numpy()
    counts_pred = counts.to_numpy()
    print("Labels for pred: ", unique_labels_pred, counts_pred, "Total:", counts_pred.sum())


    # Surandame klasifikavimo tikslumą ir išvedame rezultatus
    print("\nKlasifikavimo tikslumas")
    comment = []
    test_labels = df_matched['annot'].values
    pred_labels = df_matched['pred'].values
    print_classification_results(test_labels, pred_labels, comment)

    # Sulyginimui su binarine klasifikacija perskaičiuojame labels: N=0, S=1, V=1, U=1
    # Ensure ndarray to avoid cases wHOME a scalar bool leaks in and lacks `.astype`
    test_labels_bin = (np.asarray(test_labels) != 0).astype(int)
    pred_labels_bin = (np.asarray(pred_labels) != 0).astype(int)
    print("\nClassification results for binary case")
    evaluate_binary_classification(test_labels_bin, pred_labels_bin, positive_class=1)

"""

if __name__ == "__main__":
    main()

    