from typing import Any, Dict
from pathlib import Path
import numpy as np

from ecg_denoising_pipeline import read_df_annot
from ecg_denoising_pipeline import map_mark_denoised_to_start
from ecg_ectopy_pipeline import (
    merge_rpeaks_with_annotations,
    print_classification_results,
    evaluate_binary_classification,
)

def evaluate_ectopy_classification_against_annotations(
    *,
    res_denoising,
    res_ectopy,
    data_dir: Path | str,
    file_name: str,
    fs: int = 200,
    tolerance: int = 20,
    drop_annot_u: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare ectopy classification results against physician annotations.

    Steps
    -----
    1. Take R-peaks detected on denoised signal.
    2. Map R-peak indices back to original signal coordinates.
    3. Read physician annotations from source data.
    4. Match predicted R-peaks with annotated R-peaks using tolerance.
    5. Optionally remove rows with annot == 3 (U class).
    6. Print and return multiclass and binary classification information.

    Parameters
    ----------
    res_denoising : Any
        Result object returned by denoising pipeline.
    res_ectopy : Any
        Result object returned by ectopy pipeline.
    data_dir : Path | str
        Directory containing ECG files and annotation data.
    file_name : str
        Record/file name used to read annotations.
    fs : int, default 200
        Sampling frequency.
    tolerance : int, default 20
        Matching tolerance in samples.
    drop_annot_u : bool, default True
        If True, rows with annot == 3 are removed before evaluation.
    verbose : bool, default True
        If True, print intermediate summaries and classification metrics.

    Returns
    -------
    Dict[str, Any]
        Dictionary with mapped peaks, matched/unmatched dataframes,
        labels, and summary values.
    """
    if res_denoising is None:
        raise ValueError("res_denoising is None")
    if res_ectopy is None:
        raise ValueError("res_ectopy is None")
    if getattr(res_ectopy, "rpeaks_on_denoised_df", None) is None:
        raise ValueError("res_ectopy.rpeaks_on_denoised_df is None")

    # +++++++ RPIKŲ IR EKSTRASISTOLIŲ INDEKSŲ PERSKAIČIAVIMAS Į PRADINĮ SIGNALĄ +++++++

    rpeaks_on_denoised = (
        res_ectopy.rpeaks_on_denoised_df["rpeak"].astype(int).tolist()
    )

    rpeaks_on_start, rpeaks_on_gap = map_mark_denoised_to_start(
        rpeaks_on_denoised,
        res_denoising,
    )
    rpeaks_on_start = np.array(rpeaks_on_start, dtype=int)

    # Build dataframe in 'start' coordinates (keeps Index and pred)
    rpeaks_on_start_df = res_ectopy.rpeaks_on_denoised_df.copy()
    rpeaks_on_start_df["rpeak"] = rpeaks_on_start

    # Backward-compatible alias if needed elsewhere
    rpeaks_df_start = rpeaks_on_start_df

    if verbose:
        print(f"Mapped {len(rpeaks_on_start_df)} R-peaks from final -> start")

    # ++++++++++++++++++++++++++ ECG PŪPSNIŲ KLASIFIKACIJOS SULYGINIMAS SU ANOTACIJOMIS ++++++++++++++++++++++++++

    if verbose:
        print(f"\nfileName: {file_name}")

    annot_df = read_df_annot(data_dir, file_name)

    if verbose:
        print(f"\nlen(annot_df): {len(annot_df)}")
        print("\ntolerance:", tolerance)

    df_matched, df_unmatched, not_found_count, description = merge_rpeaks_with_annotations(
        rpeaks_on_start_df,
        annot_df,
        tolerance=tolerance,
    )

    if verbose:
        print(description)
        print(f"\nlen(df_matched): {len(df_matched)}")
        print(f"\nlen(df_unmatched): {len(df_unmatched)}")

    df_matched = df_matched.sort_values(by="diff", ascending=False).reset_index(drop=True)

    if len(df_unmatched) > 0:
        df_unmatched = df_unmatched.copy()
        df_unmatched["rpeak_sec"] = df_unmatched["rpeak"] / fs
        df_unmatched["closest_rpeak_annot_sec"] = df_unmatched["closest_rpeak_annot"] / fs

    removed_count = 0
    if drop_annot_u and "annot" in df_matched.columns:
        removed_count = int((df_matched["annot"] == 3).sum())
        df_matched = df_matched[df_matched["annot"] != 3].reset_index(drop=True)

        if verbose:
            print(f"\nRemoved {removed_count} rows with annot == 3")

    counts_annot_series = df_matched["annot"].value_counts(dropna=False)
    unique_labels_annot = counts_annot_series.index.to_numpy()
    counts_annot = counts_annot_series.to_numpy()

    counts_pred_series = df_matched["pred"].value_counts(dropna=False)
    unique_labels_pred = counts_pred_series.index.to_numpy()
    counts_pred = counts_pred_series.to_numpy()

    if verbose:
        print("\nLabels for annot: ", unique_labels_annot, counts_annot, "Total:", counts_annot.sum())
        print("Labels for pred: ", unique_labels_pred, counts_pred, "Total:", counts_pred.sum())

    # Klasifikavimo tikslumas
    comment = []
    test_labels = df_matched["annot"].values
    pred_labels = df_matched["pred"].values

    if verbose:
        print("\nKlasifikavimo tikslumas")
        print_classification_results(test_labels, pred_labels, comment)

    # Binarinė klasifikacija: N=0, S/V/U=1
    test_labels_bin = (np.asarray(test_labels) != 0).astype(int)
    pred_labels_bin = (np.asarray(pred_labels) != 0).astype(int)

    if verbose:
        print("\nClassification results for binary case")
        evaluate_binary_classification(
            test_labels_bin,
            pred_labels_bin,
            positive_class=1,
        )

    return {
        "rpeaks_on_denoised": rpeaks_on_denoised,
        "rpeaks_on_start": rpeaks_on_start,
        "rpeaks_on_gap": rpeaks_on_gap,
        "rpeaks_on_start_df": rpeaks_on_start_df,
        "rpeaks_df_start": rpeaks_df_start,
        "annot_df": annot_df,
        "df_matched": df_matched,
        "df_unmatched": df_unmatched,
        "not_found_count": not_found_count,
        "description": description,
        "removed_count": removed_count,
        "unique_labels_annot": unique_labels_annot,
        "counts_annot": counts_annot,
        "unique_labels_pred": unique_labels_pred,
        "counts_pred": counts_pred,
        "test_labels": test_labels,
        "pred_labels": pred_labels,
        "test_labels_bin": test_labels_bin,
        "pred_labels_bin": pred_labels_bin,
        "comment": comment,
    }