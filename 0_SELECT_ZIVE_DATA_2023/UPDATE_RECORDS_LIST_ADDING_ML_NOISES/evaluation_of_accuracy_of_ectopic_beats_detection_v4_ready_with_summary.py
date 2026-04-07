# https://chatgpt.com/c/69c92b31-0418-8392-805d-ef5aa6e89d50

from __future__ import annotations

from typing import Any, Dict, List
from typing import Optional, Union, cast
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum
import argparse
import time
import numpy as np
import pandas as pd


# naudosime funkcijas iš zive_data_read_utils.py
from zive_data_read_utils import load_ecg_npy, list_ecg_records, read_json_file

try:
    from ecg_denoising_pipeline import (
        ECGDenoisingPipeline,
        load_denoising_config_yaml,
        DenoisingPipelineConfig,
        resolve_model_path,
        read_df_annot_from_json
    )
except Exception as exc:
    raise ImportError(
        "Cannot import ECG denoising pipeline helpers. "
        "Update imports to match your project structure.\n"
        f"Original error: {exc}"
    ) from exc

try:
    from ecg_denoising_pipeline.steps import check_denoising_config
except Exception:
    check_denoising_config = None

from record_noise_stats import calc_noise_stats_from_denoised_result

try:
    from ecg_ectopy_pipeline import (
        ECGEctopyPipeline,
        EctopyPipelineConfig,
        EctopyPipelineResult,
        load_ectopy_config_yaml,
        merge_rpeaks_with_annotations,
        print_classification_results,
        evaluate_binary_classification,
    )
except Exception as exc:
    raise ImportError(
        "Cannot import ECG ectopy pipeline helpers. "
        "Update imports to match your project structure.\n"
        f"Original error: {exc}"
    ) from exc

# IMPORTANT:
# Adjust this import to the actual module where the function lives in your project.
try:
    from ecg_denoising_pipeline import map_mark_denoised_to_start
except Exception as exc:
    raise ImportError(
        "Cannot import map_mark_denoised_to_start. "
        "Update the import to match your project structure.\n"
        f"Original error: {exc}"
    ) from exc


# ======================================================================================
#                                   REŽIMAI
# ======================================================================================

class DenoisingMode(str, Enum):
    """
    3 galimi darbo režimai:

    FULL
        --denoising
        Įjungtas visas denoising pipeline, įskaitant motions etapą.

    NO_MOTIONS
        --denoising + --disable-motions
        Įjungtas denoising pipeline, bet motions etapas išjungtas.

    OFF
        Nėra --denoising
        Išjungiami outliers / rdropouts / motions etapai,
        tačiau pipeline vis tiek paleidžiamas, kad ectopy gautų
        tokio pat tipo input objektą kaip ir kitais režimais.
    """
    FULL = "full"
    NO_MOTIONS = "no_motions"
    OFF = "off"


@dataclass
class PipelineBundle:
    mode: DenoisingMode
    cfg_denoising: DenoisingPipelineConfig
    denoising_pipe: ECGDenoisingPipeline
    cfg_ectopy: EctopyPipelineConfig
    ectopy_pipe: ECGEctopyPipeline
    marker: str


@dataclass
class AggregateMetrics:
    """
    Kaupti globalius ir per-įrašą skaičiuojamus vertinimo rezultatus.
    """
    n_records_evaluated: int = 0
    n_records_failed_eval: int = 0

    total_matched_rows: int = 0
    total_unmatched_rows: int = 0
    total_removed_u_rows: int = 0

    # Global multiclass labels (po visų sujungimo)
    all_test_labels: List[int] = field(default_factory=list)
    all_pred_labels: List[int] = field(default_factory=list)

    # Global binary labels
    all_test_labels_bin: List[int] = field(default_factory=list)
    all_pred_labels_bin: List[int] = field(default_factory=list)

    # Per-record metrics
    per_record_accuracy: List[float] = field(default_factory=list)
    per_record_sensitivity: List[float] = field(default_factory=list)
    per_record_specificity: List[float] = field(default_factory=list)


# ======================================================================================
#                              PAGALBINĖS FUNKCIJOS
# ======================================================================================

def get_denoising_mode(denoising: bool, disable_motions: bool) -> DenoisingMode:
    if denoising and not disable_motions:
        return DenoisingMode.FULL
    if denoising and disable_motions:
        return DenoisingMode.NO_MOTIONS
    return DenoisingMode.OFF


def print_denoising_case(mode: DenoisingMode) -> None:
    if mode == DenoisingMode.FULL:
        print("CASE 3: denoising pipeline is ENABLED, including motions detection.")
    elif mode == DenoisingMode.NO_MOTIONS:
        print("CASE 2: denoising pipeline is ENABLED, but motions detection is DISABLED.")
    else:
        print("CASE 1: denoising pipeline is DISABLED.")
        print(
            "        Safe mode: the pipeline will still run with outliers, rdropouts, "
            "and motions detection disabled,"
        )
        print(
            "        so the ectopy pipeline receives the same type of input object."
        )


def mode_suffix(mode: DenoisingMode) -> str:
    if mode == DenoisingMode.FULL:
        return "with motions detection enabled"
    if mode == DenoisingMode.NO_MOTIONS:
        return "with motions detection disabled"
    return "with denoising disabled"


def safe_mean(values: List[float]) -> float:
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))


# ======================================================================================
#                           DENOISING / ECTOPY CONFIG PARUOŠIMAS
# ======================================================================================

def prepare_denoising_pipeline_cfg(
    denoising_config_path: Path,
    denoising_model_dir: Path,
    mode: DenoisingMode,
) -> DenoisingPipelineConfig:
    """
    Paruošia denoising pipeline konfigūraciją pagal pasirinktą režimą.
    """
    if not denoising_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {denoising_config_path}")

    cfg = load_denoising_config_yaml(str(denoising_config_path))

    if check_denoising_config is not None:
        check_denoising_config(cfg)

    # motions model path reikalingas config'ui,
    # net jei motions etapas vėliau išjungiamas.
    if not denoising_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {denoising_model_dir}")

    cfg.motions.model_name = resolve_model_path(
        denoising_model_dir,
        cfg.motions.model_name,
    )

    # bazinis filter etapas paliekamas įjungtas
    cfg.filter.enabled = True

    if mode == DenoisingMode.FULL:
        cfg.outliers.enabled = True
        cfg.rdropouts.enabled = True
        cfg.motions.enabled = True

    elif mode == DenoisingMode.NO_MOTIONS:
        cfg.outliers.enabled = True
        cfg.rdropouts.enabled = True
        cfg.motions.enabled = False

    else:
        # SAFE OFF variantas:
        # pipeline veikia, bet triukšmo šalinimo etapai išjungti
        cfg.outliers.enabled = False
        cfg.rdropouts.enabled = False
        cfg.motions.enabled = False

    return cfg


def create_unet_marker(cfg_denoising: DenoisingPipelineConfig) -> str:
    """
    Create a marker string based on the UNet model name and threshold
    for output column naming.
    """
    unet_model_name = Path(cfg_denoising.motions.model_name).name
    marker = unet_model_name.removeprefix("resunet_ecg").removesuffix(".keras")
    threshold = cfg_denoising.motions.threshold
    threshold_str = str(threshold).replace(".", "_")
    marker += f"_{threshold_str}"
    return marker


def prepare_ectopy_pipeline_cfg(
    ectopy_config_path: Path,
    ectopy_model_dir: Path,
) -> EctopyPipelineConfig:
    """
    Paruošia ectopy pipeline konfigūraciją.
    """
    if not ectopy_config_path.exists():
        raise FileNotFoundError(f"Ectopy config file not found: {ectopy_config_path}")
    if not ectopy_model_dir.exists():
        raise FileNotFoundError(f"Ectopy model directory not found: {ectopy_model_dir}")

    cfg: EctopyPipelineConfig = load_ectopy_config_yaml(ectopy_config_path)

    cfg.ectopy.model_name = resolve_model_path(
        ectopy_model_dir,
        cfg.ectopy.model_name,
    )
    cfg.ectopy.scaler_name = resolve_model_path(
        ectopy_model_dir,
        cfg.ectopy.scaler_name,
    )

    cfg.ectopy.ectopy_removing = False
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


# ======================================================================================
#                         KLASIFIKACIJOS VERTINIMAS PRIEŠ ANOTACIJAS
# ======================================================================================




def evaluate_ectopy_classification_against_annotations(
    *,
    res_denoising,
    res_ectopy,
    json_path: Path | str,
    fs: int = 200,
    tolerance: int = 20,
    drop_annot_u: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compare ectopy classification results against physician annotations.

    Parameters
    ----------
    res_denoising
        Result object from denoising pipeline.
    res_ectopy
        Result object from ectopy pipeline.
    json_path : Path | str
        Full path to JSON annotation file.
    fs : int
        Sampling frequency.
    tolerance : int
        Matching tolerance in samples between predicted and annotated R-peaks.
    drop_annot_u : bool
        If True, remove rows with annot == 3 (U class) before evaluation.
    verbose : bool
        If True, print detailed debug information.

    Returns
    -------
    Dict[str, Any]
        Dictionary with intermediate dataframes, labels, counts, and evaluation arrays.
    """
    if res_denoising is None:
        raise ValueError("res_denoising is None")

    if res_ectopy is None:
        raise ValueError("res_ectopy is None")

    rpeaks_df = getattr(res_ectopy, "rpeaks_on_denoised_df", None)
    if rpeaks_df is None:
        raise ValueError("res_ectopy.rpeaks_on_denoised_df is None")

    if not hasattr(rpeaks_df, "columns"):
        raise TypeError("res_ectopy.rpeaks_on_denoised_df is not a DataFrame-like object")

    if "rpeak" not in rpeaks_df.columns:
        raise ValueError(
            "res_ectopy.rpeaks_on_denoised_df does not contain required column 'rpeak'"
        )

    json_path = Path(json_path)

    if verbose:
        print("\n" + "." * 90)
        print("EVALUATION AGAINST ANNOTATIONS")
        print("." * 90)
        print(f"DEBUG json_path   : {json_path}")
        print(f"DEBUG tolerance   : {tolerance}")
        print(f"DEBUG drop_annot_u: {drop_annot_u}")

    # ------------------------------------------------------------------
    # 1. R-peaks from denoised coordinates
    # ------------------------------------------------------------------
    try:
        rpeaks_on_denoised = rpeaks_df["rpeak"].astype(int).tolist()
    except Exception as exc:
        raise ValueError(
            "Failed to extract integer 'rpeak' values from "
            "res_ectopy.rpeaks_on_denoised_df"
        ) from exc

    if verbose:
        print(f"DEBUG len(rpeaks_on_denoised): {len(rpeaks_on_denoised)}")

    # ------------------------------------------------------------------
    # 2. Map back to original signal coordinates
    # ------------------------------------------------------------------
    try:
        rpeaks_on_start, rpeaks_on_gap = map_mark_denoised_to_start(
            rpeaks_on_denoised,
            res_denoising,
        )
    except Exception as exc:
        raise RuntimeError("map_mark_denoised_to_start(...) failed") from exc

    if rpeaks_on_start is None:
        raise ValueError("map_mark_denoised_to_start returned rpeaks_on_start=None")

    rpeaks_on_start = np.asarray(rpeaks_on_start, dtype=int)

    if verbose:
        print(f"DEBUG len(rpeaks_on_start): {len(rpeaks_on_start)}")
        if rpeaks_on_gap is None:
            print("DEBUG rpeaks_on_gap: None")
        else:
            try:
                print(f"DEBUG len(rpeaks_on_gap): {len(rpeaks_on_gap)}")
            except Exception:
                print(f"DEBUG rpeaks_on_gap type: {type(rpeaks_on_gap)}")

    if len(rpeaks_on_start) != len(rpeaks_on_denoised):
        raise ValueError(
            "Mapped rpeaks length mismatch: "
            f"len(rpeaks_on_denoised)={len(rpeaks_on_denoised)}, "
            f"len(rpeaks_on_start)={len(rpeaks_on_start)}"
        )

    rpeaks_on_start_df = rpeaks_df.copy()
    rpeaks_on_start_df["rpeak"] = rpeaks_on_start

    # backward-compatible alias
    rpeaks_df_start = rpeaks_on_start_df

    if verbose:
        print(f"Mapped {len(rpeaks_on_start_df)} R-peaks from final -> start")
        print(f"DEBUG rpeaks_on_start_df columns: {list(rpeaks_on_start_df.columns)}")

    # ------------------------------------------------------------------
    # 3. Read physician annotations
    # ------------------------------------------------------------------
    if verbose:
        print(f"\nReading annotations from json_path={str(json_path)!r}")


    try:
        annot_df = read_df_annot_from_json(json_path)
    except Exception as exc:
        raise RuntimeError(
            f"read_df_annot_from_json(json_path={str(json_path)!r}) failed"
        ) from exc

    if annot_df is None:
        raise ValueError(
            "read_df_annot_from_json returned None. "
            f"Likely annotation file was not found: json_path={str(json_path)!r}"
        )

    if not hasattr(annot_df, "columns"):
        raise TypeError(
            "read_df_annot_from_json did not return a DataFrame-like object. "
            f"Returned type: {type(annot_df)}"
        )

    if verbose:
        print(f"DEBUG len(annot_df): {len(annot_df)}")
        print(f"DEBUG annot_df columns: {list(annot_df.columns)}")

    required_annot_cols = {"rpeak", "annot"}
    missing_annot_cols = [c for c in required_annot_cols if c not in annot_df.columns]
    if missing_annot_cols:
        if verbose:
            print(
                "DEBUG: annotation dataframe does not contain standard columns "
                f"{missing_annot_cols}. Continuing, because merge function may still handle it."
            )

    # ------------------------------------------------------------------
    # 4. Match predicted peaks with annotations
    # ------------------------------------------------------------------
    try:
        df_matched, df_unmatched, not_found_count, description = merge_rpeaks_with_annotations(
            rpeaks_on_start_df,
            annot_df,
            tolerance=tolerance,
        )
    except Exception as exc:
        raise RuntimeError("merge_rpeaks_with_annotations(...) failed") from exc

    if df_matched is None:
        raise ValueError("merge_rpeaks_with_annotations returned df_matched=None")

    if df_unmatched is None:
        if verbose:
            print("DEBUG: df_unmatched is None -> replacing with empty DataFrame")
        df_unmatched = pd.DataFrame()

    if description is None:
        description = "No description returned by merge_rpeaks_with_annotations."

    if verbose:
        print(description)
        print(f"\nlen(df_matched): {len(df_matched)}")
        print(f"len(df_unmatched): {len(df_unmatched)}")
        print(f"DEBUG not_found_count: {not_found_count}")

    if len(df_matched) == 0 and verbose:
        print("DEBUG: df_matched is empty. Metrics will be computed on empty arrays.")

    if "diff" in df_matched.columns:
        df_matched = df_matched.sort_values(by="diff", ascending=False).reset_index(drop=True)
    else:
        df_matched = df_matched.reset_index(drop=True)
        if verbose:
            print("DEBUG: column 'diff' not found in df_matched, skipping sort")

    if len(df_unmatched) > 0:
        df_unmatched = df_unmatched.copy()
        if "rpeak" in df_unmatched.columns:
            df_unmatched["rpeak_sec"] = df_unmatched["rpeak"] / fs
        else:
            if verbose:
                print("DEBUG: 'rpeak' column not found in df_unmatched, cannot create rpeak_sec")

        if "closest_rpeak_annot" in df_unmatched.columns:
            df_unmatched["closest_rpeak_annot_sec"] = df_unmatched["closest_rpeak_annot"] / fs
        else:
            if verbose:
                print(
                    "DEBUG: 'closest_rpeak_annot' column not found in df_unmatched, "
                    "cannot create closest_rpeak_annot_sec"
                )

    # ------------------------------------------------------------------
    # 5. Optionally remove U class from evaluation
    # ------------------------------------------------------------------
    removed_count = 0
    if drop_annot_u:
        if "annot" not in df_matched.columns:
            raise ValueError(
                "drop_annot_u=True, but df_matched does not contain column 'annot'"
            )

        removed_count = int((df_matched["annot"] == 3).sum())
        df_matched = df_matched[df_matched["annot"] != 3].reset_index(drop=True)

        if verbose:
            print(f"\nRemoved {removed_count} rows with annot == 3")

    # ------------------------------------------------------------------
    # 6. Label counts
    # ------------------------------------------------------------------
    if "annot" not in df_matched.columns:
        raise ValueError("df_matched does not contain required column 'annot'")

    if "pred" not in df_matched.columns:
        raise ValueError("df_matched does not contain required column 'pred'")

    counts_annot_series = df_matched["annot"].value_counts(dropna=False)
    unique_labels_annot = counts_annot_series.index.to_numpy()
    counts_annot = counts_annot_series.to_numpy()

    counts_pred_series = df_matched["pred"].value_counts(dropna=False)
    unique_labels_pred = counts_pred_series.index.to_numpy()
    counts_pred = counts_pred_series.to_numpy()

    if verbose:
        print("\nLabels for annot: ", unique_labels_annot, counts_annot, "Total:", counts_annot.sum())
        print("Labels for pred: ", unique_labels_pred, counts_pred, "Total:", counts_pred.sum())

    # ------------------------------------------------------------------
    # 7. Multiclass classification
    # ------------------------------------------------------------------
    test_labels = np.asarray(df_matched["annot"].values)
    pred_labels = np.asarray(df_matched["pred"].values)

    comment = []

    if verbose:
        print("\nKlasifikavimo tikslumas")
        if len(test_labels) == 0:
            print("DEBUG: no matched rows after filtering; multiclass metrics skipped for this record")
        else:
            try:
                print_classification_results(test_labels, pred_labels, comment)
            except Exception as exc:
                print(f"DEBUG: print_classification_results(...) failed: {exc}")

    # ------------------------------------------------------------------
    # 8. Binary classification: N=0, S/V/U=1
    # ------------------------------------------------------------------
    test_labels_bin = (np.asarray(test_labels) != 0).astype(int)
    pred_labels_bin = (np.asarray(pred_labels) != 0).astype(int)

    if verbose:
        print("\nClassification results for binary case")
        if len(test_labels_bin) == 0:
            print("DEBUG: no matched rows after filtering; binary metrics skipped for this record")
        else:
            try:
                evaluate_binary_classification(
                    test_labels_bin,
                    pred_labels_bin,
                    positive_class=1,
                )
            except Exception as exc:
                print(f"DEBUG: evaluate_binary_classification(...) failed: {exc}")

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

# ======================================================================================
#                               AGREGUOTOS METRIKOS
# ======================================================================================

def compute_binary_confusion_and_metrics(
    test_labels_bin: np.ndarray,
    pred_labels_bin: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary confusion matrix and basic metrics.

    Positive class:
        1 = ectopic (S/V/U)
    Negative class:
        0 = normal (N)
    """
    test_labels_bin = np.asarray(test_labels_bin).astype(int)
    pred_labels_bin = np.asarray(pred_labels_bin).astype(int)

    tp = int(((test_labels_bin == 1) & (pred_labels_bin == 1)).sum())
    tn = int(((test_labels_bin == 0) & (pred_labels_bin == 0)).sum())
    fp = int(((test_labels_bin == 0) & (pred_labels_bin == 1)).sum())
    fn = int(((test_labels_bin == 1) & (pred_labels_bin == 0)).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "n": total,
    }


def build_multiclass_confusion_matrix(
    test_labels: np.ndarray,
    pred_labels: np.ndarray,
    labels: List[int] | None = None,
) -> Dict[str, Any]:
    """
    Build multiclass confusion matrix without sklearn.
    """
    test_labels = np.asarray(test_labels)
    pred_labels = np.asarray(pred_labels)

    if labels is None:
        labels = sorted(set(test_labels.tolist()) | set(pred_labels.tolist()))

    labels_arr = np.asarray(labels)
    idx = {lab: i for i, lab in enumerate(labels_arr)}
    cm = np.zeros((len(labels_arr), len(labels_arr)), dtype=int)

    for t, p in zip(test_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    return {
        "labels": labels_arr,
        "matrix": cm,
    }


def compute_multiclass_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    """
    Compute multiclass metrics from confusion matrix.

    Returns
    -------
    Dict[str, Any]
        accuracy
        precision_macro
        recall_macro
        f1_macro
        precision_weighted
        recall_weighted
        f1_weighted
        per_class_precision
        per_class_recall
        per_class_f1
        support
    """
    cm = np.asarray(cm, dtype=float)

    total = cm.sum()
    correct = np.trace(cm)
    accuracy = correct / total if total > 0 else float("nan")

    n_classes = cm.shape[0]
    per_class_precision = np.full(n_classes, np.nan, dtype=float)
    per_class_recall = np.full(n_classes, np.nan, dtype=float)
    per_class_f1 = np.full(n_classes, np.nan, dtype=float)
    support = cm.sum(axis=1)

    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if not np.isnan(precision) and not np.isnan(recall) and (precision + recall) > 0
            else np.nan
        )

        per_class_precision[i] = precision
        per_class_recall[i] = recall
        per_class_f1[i] = f1

    precision_macro = np.nanmean(per_class_precision) if n_classes > 0 else float("nan")
    recall_macro = np.nanmean(per_class_recall) if n_classes > 0 else float("nan")
    f1_macro = np.nanmean(per_class_f1) if n_classes > 0 else float("nan")

    support_sum = support.sum()
    if support_sum > 0:
        precision_weighted = np.nansum(per_class_precision * support) / support_sum
        recall_weighted = np.nansum(per_class_recall * support) / support_sum
        f1_weighted = np.nansum(per_class_f1 * support) / support_sum
    else:
        precision_weighted = float("nan")
        recall_weighted = float("nan")
        f1_weighted = float("nan")

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "support": support,
    }


def update_aggregate_metrics(
    agg: AggregateMetrics,
    eval_res: Dict[str, Any],
) -> None:
    """
    Update global accumulator with one record's evaluation results.
    """
    df_matched = eval_res["df_matched"]
    df_unmatched = eval_res["df_unmatched"]
    removed_count = int(eval_res["removed_count"])

    test_labels = np.asarray(eval_res["test_labels"]).astype(int)
    pred_labels = np.asarray(eval_res["pred_labels"]).astype(int)
    test_labels_bin = np.asarray(eval_res["test_labels_bin"]).astype(int)
    pred_labels_bin = np.asarray(eval_res["pred_labels_bin"]).astype(int)

    agg.n_records_evaluated += 1
    agg.total_matched_rows += int(len(df_matched))
    agg.total_unmatched_rows += int(len(df_unmatched))
    agg.total_removed_u_rows += removed_count

    agg.all_test_labels.extend(test_labels.tolist())
    agg.all_pred_labels.extend(pred_labels.tolist())
    agg.all_test_labels_bin.extend(test_labels_bin.tolist())
    agg.all_pred_labels_bin.extend(pred_labels_bin.tolist())

    binary_metrics = compute_binary_confusion_and_metrics(
        test_labels_bin=test_labels_bin,
        pred_labels_bin=pred_labels_bin,
    )

    agg.per_record_accuracy.append(binary_metrics["accuracy"])
    agg.per_record_sensitivity.append(binary_metrics["sensitivity"])
    agg.per_record_specificity.append(binary_metrics["specificity"])


def fmt_float(x: float) -> str:
    if x is None:
        return "nan"
    try:
        if np.isnan(x):
            return "nan"
    except Exception:
        pass
    return f"{x:.4f}"


def build_aggregate_report_text(
    agg: AggregateMetrics,
    global_binary_metrics: bool = False,
) -> str:
    """
    Build final global and averaged evaluation report as a clean text block.
    """
    lines: List[str] = []

    lines.append("")
    lines.append("=" * 90)
    lines.append("BENDRA VERTINIMO SUVESTINĖ")
    lines.append("=" * 90)

    lines.append(f"Evaluated records          : {agg.n_records_evaluated}")
    lines.append(f"Failed evaluation records  : {agg.n_records_failed_eval}")
    lines.append(f"Total matched rows         : {agg.total_matched_rows}")
    lines.append(f"Total unmatched rows       : {agg.total_unmatched_rows}")
    lines.append(f"Total removed annot==3     : {agg.total_removed_u_rows}")

    if global_binary_metrics:
        global_binary = compute_binary_confusion_and_metrics(
            test_labels_bin=np.asarray(agg.all_test_labels_bin, dtype=int),
            pred_labels_bin=np.asarray(agg.all_pred_labels_bin, dtype=int),
        )

        lines.append("")
        lines.append("GLOBAL BINARY CONFUSION MATRIX")
        lines.append("Rows = true labels, Cols = predicted labels")
        lines.append("          Pred 0     Pred 1")
        lines.append(f"True 0    {global_binary['tn']:8d}   {global_binary['fp']:8d}")
        lines.append(f"True 1    {global_binary['fn']:8d}   {global_binary['tp']:8d}")

        lines.append("")
        lines.append("GLOBAL BINARY METRICS")
        lines.append(f"Accuracy    : {fmt_float(global_binary['accuracy'])}")
        lines.append(f"Sensitivity : {fmt_float(global_binary['sensitivity'])}")
        lines.append(f"Specificity : {fmt_float(global_binary['specificity'])}")
        lines.append(f"Precision   : {fmt_float(global_binary['precision'])}")

    mean_acc = safe_mean(agg.per_record_accuracy)
    mean_sens = safe_mean(agg.per_record_sensitivity)
    mean_spec = safe_mean(agg.per_record_specificity)

    lines.append("")
    lines.append("MEAN PER-RECORD BINARY METRICS")
    lines.append(f"Mean accuracy    : {fmt_float(mean_acc)}")
    lines.append(f"Mean sensitivity : {fmt_float(mean_sens)}")
    lines.append(f"Mean specificity : {fmt_float(mean_spec)}")

    multiclass = build_multiclass_confusion_matrix(
        test_labels=np.asarray(agg.all_test_labels, dtype=int),
        pred_labels=np.asarray(agg.all_pred_labels, dtype=int),
        labels=[0, 1, 2],
    )
    labels = multiclass["labels"]
    cm = multiclass["matrix"]
    multiclass_metrics = compute_multiclass_metrics_from_cm(cm)

    lines.append("")
    lines.append("GLOBAL MULTICLASS SUMMARY")
    lines.append(f"Accuracy:  {fmt_float(multiclass_metrics['accuracy'])}")
    lines.append(f"Precision: {fmt_float(multiclass_metrics['precision_macro'])}")
    lines.append(f"Recall:    {fmt_float(multiclass_metrics['recall_macro'])}")
    lines.append(f"F1-score:  {fmt_float(multiclass_metrics['f1_macro'])}")

    lines.append("")
    lines.append("Confusion Matrix:")
    lines.append(str(cm))

    lines.append("")
    lines.append("GLOBAL MULTICLASS CONFUSION MATRIX")
    lines.append(f"Labels: {labels.tolist()}")
    lines.append("Rows = true labels, Cols = predicted labels")
    lines.append(str(cm))

    lines.append("")
    lines.append("GLOBAL MULTICLASS METRICS")
    lines.append(f"Accuracy            : {fmt_float(multiclass_metrics['accuracy'])}")
    lines.append(f"Precision (macro)   : {fmt_float(multiclass_metrics['precision_macro'])}")
    lines.append(f"Recall (macro)      : {fmt_float(multiclass_metrics['recall_macro'])}")
    lines.append(f"F1-score (macro)    : {fmt_float(multiclass_metrics['f1_macro'])}")
    lines.append(f"Precision (weighted): {fmt_float(multiclass_metrics['precision_weighted'])}")
    lines.append(f"Recall (weighted)   : {fmt_float(multiclass_metrics['recall_weighted'])}")
    lines.append(f"F1-score (weighted) : {fmt_float(multiclass_metrics['f1_weighted'])}")

    lines.append("")
    lines.append("PER-CLASS MULTICLASS METRICS")
    for i, lab in enumerate(labels):
        lines.append(
            f"Class {lab}: "
            f"support={int(multiclass_metrics['support'][i])}, "
            f"precision={fmt_float(multiclass_metrics['per_class_precision'][i])}, "
            f"recall={fmt_float(multiclass_metrics['per_class_recall'][i])}, "
            f"f1={fmt_float(multiclass_metrics['per_class_f1'][i])}"
        )

    return "\n".join(lines)


def print_aggregate_report(
    agg: AggregateMetrics,
    global_binary_metrics: bool = False,
) -> str:
    """
    Print final global and averaged evaluation report and also return it as text.
    """
    report_text = build_aggregate_report_text(
        agg=agg,
        global_binary_metrics=global_binary_metrics,
    )
    print(report_text)
    return report_text


def write_summary_text(summary_path: Path, report_text: str) -> None:
    summary_path = Path(summary_path).expanduser().resolve()
    if summary_path.parent and not summary_path.parent.exists():
        summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report_text.rstrip() + "\n")

    print(f"\nClean summary written to: {summary_path}")


# ======================================================================================
#                               PIPELINE PARUOŠIMAS
# ======================================================================================

def prepare_pipelines(args: argparse.Namespace) -> PipelineBundle:
    """
    Paruošia pipeline objektus pagal pasirinktą darbo režimą.
    """
    mode = get_denoising_mode(
        denoising=args.denoising,
        disable_motions=args.disable_motions,
    )

    print("\n" + "=" * 90)
    print("PIPELINE PARUOŠIMAS")
    print("=" * 90)

    print(f"\nDenoising config path   : {args.cfg_denoising}")
    print(f"Denoising model dir     : {args.unet_model_dir}")
    print(f"Ectopy config path      : {args.cfg_ectopy}")
    print(f"Ectopy model dir        : {args.ectopy_model_dir}")
    print_denoising_case(mode)

    print("\n***** Denoising pipeline config:")
    cfg_denoising = prepare_denoising_pipeline_cfg(
        denoising_config_path=args.cfg_denoising,
        denoising_model_dir=args.unet_model_dir,
        mode=mode,
    )
    marker = create_unet_marker(cfg_denoising)
    print(f"\nUNet marker: {marker}\n")

    denoising_pipe = ECGDenoisingPipeline(cfg_denoising)

    print("\n***** Ectopy pipeline config:")
    cfg_ectopy = prepare_ectopy_pipeline_cfg(
        ectopy_config_path=args.cfg_ectopy,
        ectopy_model_dir=args.ectopy_model_dir,
    )
    ectopy_pipe = ECGEctopyPipeline(cfg_ectopy)

    return PipelineBundle(
        mode=mode,
        cfg_denoising=cfg_denoising,
        denoising_pipe=denoising_pipe,
        cfg_ectopy=cfg_ectopy,
        ectopy_pipe=ectopy_pipe,
        marker=marker,
    )


# ======================================================================================
#                             VIENO ĮRAŠO APDOROJIMAS
# ======================================================================================

def process_record(
    rec,
    bundle: PipelineBundle,
    fs: int,
) -> Dict[str, Any]:
    """
    Apdoroja vieną įrašą:
    - nuskaito signalą
    - paleidžia denoising pipeline pagal pasirinktą režimą
    - apskaičiuoja noise stats
    - paleidžia ectopy pipeline
    - grąžina rezultatus ir statistiką
    """
    print_denoising_case(bundle.mode)

    x = load_ecg_npy(rec.ecg_path)

    # SAFE variantas: visais atvejais paleidžiame denoising_pipe,
    # tik OFF režime jo etapai išjungti config'e.
    res_denoising = bundle.denoising_pipe.run(x, gaps_indices=[])

    noise_stats = calc_noise_stats_from_denoised_result(res_denoising) or {}

    print(
        f"Noise stats for {rec.ecg_path.name}: "
        f"out={noise_stats.get('out')}, "
        f"rdr={noise_stats.get('rdr')}, "
        f"mra={noise_stats.get('mra')}, "
        f"tp_pct={noise_stats.get('tp_pct', 0.0):.1f} "
        f"{mode_suffix(bundle.mode)}"
    )
    print()

    res_ectopy = bundle.ectopy_pipe.run(
        res_denoising,
        fs=fs,
        file_name=rec.ecg_path.name,
    )
    ectopy_stats = compute_ectopy_stats(res_ectopy)
    print(f"ectopy_stats={ectopy_stats} {mode_suffix(bundle.mode)}")

    return {
        "signal": x,
        "res_denoising": res_denoising,
        "noise_stats": noise_stats,
        "res_ectopy": res_ectopy,
        "ectopy_stats": ectopy_stats,
    }



# ======================================================================================
#                                        CLI
# ======================================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate the accuracy of ectopic beat detection on ECG records "
            "with optional denoising and optional motions detection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "--dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing ECG files and corresponding JSON metadata files.",
    )
    ap.add_argument(
        "--exclude-list",
        "--exclude_files",
        dest="exclude_list",
        type=Path,
        metavar="FILE",
        help=(
            "Optional text file with ECG basenames to exclude, one basename per line "
            "(without extensions)."
        ),
    )
    ap.add_argument(
        "--cfg-denoising",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to denoising pipeline YAML config.",
    )
    ap.add_argument(
        "--unet-model-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing denoising / motions model files.",
    )
    ap.add_argument(
        "--cfg-ectopy",
        type=Path,
        required=True,
        metavar="FILE",
        help="Path to ectopy pipeline YAML config.",
    )
    ap.add_argument(
        "--ectopy-model-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing ectopy model and scaler files.",
    )
    ap.add_argument(
        "--fs",
        type=int,
        default=200,
        metavar="HZ",
        help="Sampling frequency in Hz.",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=20,
        metavar="SAMPLES",
        help="Tolerance in samples for matching predicted and annotated R-peaks.",
    )
    ap.add_argument(
        "--denoising",
        action="store_true",
        help="Enable the denoising pipeline.",
    )
    ap.add_argument(
        "--disable-motions",
        action="store_true",
        help=(
            "Disable only the motions detection stage inside the denoising pipeline. "
            "This flag has effect only when --denoising is enabled."
        ),
    )
    ap.add_argument(
        "--keep-u-class",
        action="store_true",
        help="Keep annotation class U (annot == 3) in evaluation.",
    )
    ap.add_argument(
        "--all-records",
        action="store_true",
        help="Process all matched records. By default only the first 5 are processed.",
    )
    ap.add_argument(
        "--global-binary-metrics",
        action="store_true",
        help="Print global binary confusion matrix and binary classification metrics.",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose diagnostic output during evaluation.",
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Optional path to a clean summary text file.",
    )

    return ap


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args.dir = Path(args.dir).expanduser().resolve()
    args.cfg_denoising = Path(args.cfg_denoising).expanduser().resolve()
    args.unet_model_dir = Path(args.unet_model_dir).expanduser().resolve()
    args.cfg_ectopy = Path(args.cfg_ectopy).expanduser().resolve()
    args.ectopy_model_dir = Path(args.ectopy_model_dir).expanduser().resolve()

    if args.exclude_list is not None:
        args.exclude_list = Path(args.exclude_list).expanduser().resolve()

    if args.summary_out is not None:
        args.summary_out = Path(args.summary_out).expanduser().resolve()

    if not args.dir.exists() or not args.dir.is_dir():
        parser.error(f"--dir must be an existing directory: {args.dir}")

    if args.exclude_list is not None and not args.exclude_list.exists():
        parser.error(f"--exclude-list file not found: {args.exclude_list}")

    if not args.cfg_denoising.exists() or not args.cfg_denoising.is_file():
        parser.error(f"--cfg-denoising must be an existing file: {args.cfg_denoising}")

    if not args.unet_model_dir.exists() or not args.unet_model_dir.is_dir():
        parser.error(f"--unet-model-dir must be an existing directory: {args.unet_model_dir}")

    if not args.cfg_ectopy.exists() or not args.cfg_ectopy.is_file():
        parser.error(f"--cfg-ectopy must be an existing file: {args.cfg_ectopy}")

    if not args.ectopy_model_dir.exists() or not args.ectopy_model_dir.is_dir():
        parser.error(f"--ectopy-model-dir must be an existing directory: {args.ectopy_model_dir}")

    if args.fs <= 0:
        parser.error(f"--fs must be > 0. Got: {args.fs}")

    if args.tolerance < 0:
        parser.error(f"--tolerance must be >= 0. Got: {args.tolerance}")

    if args.disable_motions and not args.denoising:
        print(
            "WARNING: --disable-motions was provided without --denoising. "
            "It will have no effect because denoising mode is OFF."
        )

    return args


def format_elapsed_hhmm(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    return f"{hours:02d}:{minutes:02d}"


def format_elapsed_minutes(seconds: float) -> str:
    minutes = seconds / 60.0
    return f"{minutes:.1f} min"



def build_summary_header_text(
    args: argparse.Namespace,
    mode: DenoisingMode,
    scan_summary: Any,
    n_records_returned: int,
) -> str:
    """
    Build the initial parameter block for the clean summary file.
    """
    lines: List[str] = []

    lines.append("ECTOPIC BEAT DETECTION EVALUATION")
    lines.append("=" * 90)
    lines.append(f"Input directory          : {args.dir}")
    lines.append(f"Exclude list             : {args.exclude_list}")
    lines.append(f"Denoising config         : {args.cfg_denoising}")
    lines.append(f"UNet model directory     : {args.unet_model_dir}")
    lines.append(f"Ectopy config            : {args.cfg_ectopy}")
    lines.append(f"Ectopy model directory   : {args.ectopy_model_dir}")
    lines.append(f"Sampling frequency       : {args.fs} Hz")
    lines.append(f"Matching tolerance       : {args.tolerance} samples")
    lines.append(f"Keep U class             : {args.keep_u_class}")
    lines.append(f"Process all records      : {args.all_records}")
    lines.append(f"Quiet mode               : {args.quiet}")
    lines.append(f"Global binary metrics    : {args.global_binary_metrics}")
    lines.append(f"Summary output           : {args.summary_out}")
    lines.append(f"Denoising mode           : {mode.value}")

    if mode == DenoisingMode.FULL:
        lines.append("CASE 3: denoising pipeline is ENABLED, including motions detection.")
    elif mode == DenoisingMode.NO_MOTIONS:
        lines.append("CASE 2: denoising pipeline is ENABLED, but motions detection is DISABLED.")
    else:
        lines.append("CASE 1: denoising pipeline is DISABLED.")
        lines.append(
            "        Safe mode: the pipeline will still run with outliers, rdropouts, "
            "and motions detection disabled,"
        )
        lines.append(
            "        so the ectopy pipeline receives the same type of input object."
        )

    lines.append("=" * 90)
    lines.append(f"total_json       : {scan_summary.total_json}")
    lines.append(f"excluded         : {scan_summary.excluded}")
    lines.append(f"matched          : {scan_summary.matched}")
    lines.append(f"unmatched_json   : {scan_summary.unmatched_json}")
    lines.append(f"records returned : {n_records_returned}")

    return "\n".join(lines)

# ======================================================================================
#                                        MAIN
# ======================================================================================

def main() -> None:
    parser = build_arg_parser()
    args = validate_args(parser.parse_args(), parser)

    mode = get_denoising_mode(args.denoising, args.disable_motions)

    print("=" * 90)
    print("ECTOPIC BEAT DETECTION EVALUATION")
    print("=" * 90)
    print(f"Input directory          : {args.dir}")
    print(f"Exclude list             : {args.exclude_list}")
    print(f"Denoising config         : {args.cfg_denoising}")
    print(f"UNet model directory     : {args.unet_model_dir}")
    print(f"Ectopy config            : {args.cfg_ectopy}")
    print(f"Ectopy model directory   : {args.ectopy_model_dir}")
    print(f"Sampling frequency       : {args.fs} Hz")
    print(f"Matching tolerance       : {args.tolerance} samples")
    print(f"Keep U class             : {args.keep_u_class}")
    print(f"Process all records      : {args.all_records}")
    print(f"Quiet mode               : {args.quiet}")
    print(f"Global binary metrics    : {args.global_binary_metrics}")
    print(f"Summary output           : {args.summary_out}")
    print(f"Denoising mode           : {mode.value}")
    print_denoising_case(mode)
    print("=" * 90)

    src = args.dir

    scan_result = list_ecg_records(
        folder=src,
        data_format="auto",
        exclude_list=args.exclude_list,
    )

    records = scan_result.records
    summary = scan_result.summary

    print(f"total_json       : {summary.total_json}")
    print(f"excluded         : {summary.excluded}")
    print(f"matched          : {summary.matched}")
    print(f"unmatched_json   : {summary.unmatched_json}")
    print(f"records returned : {len(records)}")

    summary_header_text = build_summary_header_text(
        args=args,
        mode=mode,
        scan_summary=summary,
        n_records_returned=len(records),
    )

    bundle = prepare_pipelines(args)
    agg = AggregateMetrics()

    records_to_process = records if args.all_records else records[:5]

    print(f"\nFound {len(records)} matched records")
    print(f"Processing {len(records_to_process)} record(s)")

    total_cycle_start = time.perf_counter()
    record_nr = 0

    for rec in records_to_process:
        print("\n" + "-" * 90)
        record_nr += 1

        elapsed_from_start_s = time.perf_counter() - total_cycle_start
        elapsed_from_start_min = format_elapsed_minutes(elapsed_from_start_s)

        if rec.ecg_path is not None:
            print(
                f"{record_nr}/{len(records_to_process)} | "
                f"{rec.basename} | {rec.ecg_path.name} | {rec.json_path.name} | "
                f"elapsed: {elapsed_from_start_min}",
                flush=True,
            )
        else:
            print(
                f"{record_nr}/{len(records_to_process)} | "
                f"{rec.basename} | <missing ecg> | {rec.json_path.name} | "
                f"elapsed: {elapsed_from_start_min}",
                flush=True,
            )

        if rec.ecg_path is None:
            msg = f"No matching ECG file for JSON '{rec.json_path.name}'"
            print(msg)
            continue

        metadata: Dict[str, Any] = read_json_file(rec.json_path)

        try:
            results = process_record(
                rec=rec,
                bundle=bundle,
                fs=args.fs,
            )

            noise_stats = results["noise_stats"]
            ectopy_stats = results["ectopy_stats"]
            res_denoising = results["res_denoising"]
            res_ectopy = results["res_ectopy"]

            eval_res = evaluate_ectopy_classification_against_annotations(
                res_denoising=res_denoising,
                res_ectopy=res_ectopy,
                json_path=rec.json_path,
                fs=args.fs,
                tolerance=args.tolerance,
                drop_annot_u=not args.keep_u_class,
                verbose=not args.quiet,
            )

            df_matched = eval_res["df_matched"]
            df_unmatched = eval_res["df_unmatched"]

            update_aggregate_metrics(agg, eval_res)

            _ = metadata, noise_stats, ectopy_stats, df_matched, df_unmatched

        except Exception as exc:
            agg.n_records_failed_eval += 1
            print(f"WARN: failed processing for {rec.ecg_path.name}: {exc}")
            continue

    total_cycle_elapsed_s = time.perf_counter() - total_cycle_start
    print("\n" + "=" * 90)
    print(f"Total cycle time: {format_elapsed_hhmm(total_cycle_elapsed_s)} (hh:mm)")
    print("=" * 90)

    report_text = print_aggregate_report(
        agg,
        global_binary_metrics=args.global_binary_metrics,
    )

    full_summary_text = summary_header_text + "\n" + report_text

    if args.summary_out is not None:
        write_summary_text(args.summary_out, full_summary_text)


if __name__ == "__main__":
    main()
    
# def main() -> None:
#     parser = build_arg_parser()
#     args = validate_args(parser.parse_args(), parser)

#     mode = get_denoising_mode(args.denoising, args.disable_motions)

#     print("=" * 90)
#     print("ECTOPIC BEAT DETECTION EVALUATION")
#     print("=" * 90)
#     print(f"Input directory          : {args.dir}")
#     print(f"Exclude list             : {args.exclude_list}")
#     print(f"Denoising config         : {args.cfg_denoising}")
#     print(f"UNet model directory     : {args.unet_model_dir}")
#     print(f"Ectopy config            : {args.cfg_ectopy}")
#     print(f"Ectopy model directory   : {args.ectopy_model_dir}")
#     print(f"Sampling frequency       : {args.fs} Hz")
#     print(f"Matching tolerance       : {args.tolerance} samples")
#     print(f"Keep U class             : {args.keep_u_class}")
#     print(f"Process all records      : {args.all_records}")
#     print(f"Quiet mode               : {args.quiet}")
#     print(f"Global binary metrics    : {args.global_binary_metrics}")
#     print(f"Summary output           : {args.summary_out}")
#     print(f"Denoising mode           : {mode.value}")
#     print_denoising_case(mode)
#     print("=" * 90)

#     src = args.dir

#     scan_result = list_ecg_records(
#         folder=src,
#         data_format="auto",
#         exclude_list=args.exclude_list,
#     )

#     records = scan_result.records
#     summary = scan_result.summary

#     print(f"total_json       : {summary.total_json}")
#     print(f"excluded         : {summary.excluded}")
#     print(f"matched          : {summary.matched}")
#     print(f"unmatched_json   : {summary.unmatched_json}")
#     print(f"records returned : {len(records)}")

#     bundle = prepare_pipelines(args)
#     agg = AggregateMetrics()

#     records_to_process = records if args.all_records else records[:5]

#     print(f"\nFound {len(records)} matched records")
#     print(f"Processing {len(records_to_process)} record(s)")

#     total_cycle_start = time.perf_counter()
#     record_nr = 0

#     for rec in records_to_process:
#         print("\n" + "-" * 90)
#         record_nr += 1

#         elapsed_from_start_s = time.perf_counter() - total_cycle_start
#         elapsed_from_start_min = format_elapsed_minutes(elapsed_from_start_s)

#         if rec.ecg_path is not None:
#             print(
#                 f"{record_nr}/{len(records_to_process)} | "
#                 f"{rec.basename} | {rec.ecg_path.name} | {rec.json_path.name} | "
#                 f"elapsed: {elapsed_from_start_min}"
#             )
#         else:
#             print(
#                 f"{record_nr}/{len(records_to_process)} | "
#                 f"{rec.basename} | <missing ecg> | {rec.json_path.name} | "
#                 f"elapsed: {elapsed_from_start_min}"
#             )

#         if rec.ecg_path is None:
#             msg = f"No matching ECG file for JSON '{rec.json_path.name}'"
#             print(msg)
#             continue

#         metadata: Dict[str, Any] = read_json_file(rec.json_path)

#         try:
#             results = process_record(
#                 rec=rec,
#                 bundle=bundle,
#                 fs=args.fs,
#             )

#             noise_stats = results["noise_stats"]
#             ectopy_stats = results["ectopy_stats"]
#             res_denoising = results["res_denoising"]
#             res_ectopy = results["res_ectopy"]

#             eval_res = evaluate_ectopy_classification_against_annotations(
#                 res_denoising=res_denoising,
#                 res_ectopy=res_ectopy,
#                 json_path=rec.json_path,
#                 fs=args.fs,
#                 tolerance=args.tolerance,
#                 drop_annot_u=not args.keep_u_class,
#                 verbose=not args.quiet,
#             )

#             df_matched = eval_res["df_matched"]
#             df_unmatched = eval_res["df_unmatched"]

#             update_aggregate_metrics(agg, eval_res)

#             _ = metadata, noise_stats, ectopy_stats, df_matched, df_unmatched

#         except Exception as exc:
#             agg.n_records_failed_eval += 1
#             print(f"WARN: failed processing for {rec.ecg_path.name}: {exc}")
#             continue

#     total_cycle_elapsed_s = time.perf_counter() - total_cycle_start
#     print("\n" + "=" * 90)
#     print(f"Total cycle time: {format_elapsed_hhmm(total_cycle_elapsed_s)} (hh:mm)")
#     print("=" * 90)

#     report_text = print_aggregate_report(
#         agg,
#         global_binary_metrics=args.global_binary_metrics,
#     )

#     if args.summary_out is not None:
#         write_summary_text(args.summary_out, report_text)


# if __name__ == "__main__":
#     main()
