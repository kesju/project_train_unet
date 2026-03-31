# https://chatgpt.com/c/69c92b31-0418-8392-805d-ef5aa6e89d50

from __future__ import annotations

from typing import Any, Dict, List
from typing import Optional, Union, cast
from pathlib import Path
import json
from dataclasses import dataclass, field
from enum import Enum
import argparse
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
        print("CASE 1: denoising pipeline is ENABLED, including motions stage.")
    elif mode == DenoisingMode.NO_MOTIONS:
        print("CASE 2: denoising pipeline is ENABLED, but motions stage is DISABLED.")
    else:
        print("CASE 3: denoising pipeline is DISABLED.")
        print(
            "        Safe mode: pipeline will still run with outliers/rdropouts/motions disabled,"
        )
        print(
            "        so ectopy pipeline receives the same type of input object."
        )


def mode_suffix(mode: DenoisingMode) -> str:
    if mode == DenoisingMode.FULL:
        return "with enabled detecting motions"
    if mode == DenoisingMode.NO_MOTIONS:
        return "with disabled detecting motions"
    return "with disabled denoising"


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


def print_aggregate_report(
    agg: AggregateMetrics,
    global_binary_metrics: bool = False,
) -> None:
    """
    Print final global and averaged evaluation report.
    """
    print("\n" + "=" * 90)
    print("BENDRA VERTINIMO SUVESTINĖ")
    print("=" * 90)

    print(f"Evaluated records          : {agg.n_records_evaluated}")
    print(f"Failed evaluation records  : {agg.n_records_failed_eval}")
    print(f"Total matched rows         : {agg.total_matched_rows}")
    print(f"Total unmatched rows       : {agg.total_unmatched_rows}")
    print(f"Total removed annot==3     : {agg.total_removed_u_rows}")

    # ---------------------------
    # Global binary metrics
    # ---------------------------
    if global_binary_metrics:
        global_binary = compute_binary_confusion_and_metrics(
            test_labels_bin=np.asarray(agg.all_test_labels_bin, dtype=int),
            pred_labels_bin=np.asarray(agg.all_pred_labels_bin, dtype=int),
        )

        print("\nGLOBAL BINARY CONFUSION MATRIX")
        print("Rows = true labels, Cols = predicted labels")
        print("          Pred 0     Pred 1")
        print(f"True 0    {global_binary['tn']:8d}   {global_binary['fp']:8d}")
        print(f"True 1    {global_binary['fn']:8d}   {global_binary['tp']:8d}")

        print("\nGLOBAL BINARY METRICS")
        print(f"Accuracy    : {global_binary['accuracy']:.4f}")
        print(f"Sensitivity : {global_binary['sensitivity']:.4f}")
        print(f"Specificity : {global_binary['specificity']:.4f}")
        print(f"Precision   : {global_binary['precision']:.4f}")

    # ---------------------------
    # Mean per-record binary metrics
    # ---------------------------
    mean_acc = safe_mean(agg.per_record_accuracy)
    mean_sens = safe_mean(agg.per_record_sensitivity)
    mean_spec = safe_mean(agg.per_record_specificity)

    print("\nMEAN PER-RECORD BINARY METRICS")
    print(f"Mean accuracy    : {mean_acc:.4f}")
    print(f"Mean sensitivity : {mean_sens:.4f}")
    print(f"Mean specificity : {mean_spec:.4f}")

    # ---------------------------
    # Global multiclass confusion matrix
    # ---------------------------
    multiclass = build_multiclass_confusion_matrix(
        test_labels=np.asarray(agg.all_test_labels, dtype=int),
        pred_labels=np.asarray(agg.all_pred_labels, dtype=int),
        labels=[0, 1, 2],   # po annot==3 pašalinimo dažniausiai lieka N,S,V
    )
    labels = multiclass["labels"]
    cm = multiclass["matrix"]

    print("\nGLOBAL MULTICLASS SUMMARY")
    multiclass_metrics = compute_multiclass_metrics_from_cm(cm)
    print(f"Accuracy:  {multiclass_metrics['accuracy']:.4f}")
    print(f"Precision: {multiclass_metrics['precision_macro']:.4f}")
    print(f"Recall:    {multiclass_metrics['recall_macro']:.4f}")
    print(f"F1-score:  {multiclass_metrics['f1_macro']:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nGLOBAL MULTICLASS CONFUSION MATRIX")
    print("Labels:", labels.tolist())
    print("Rows = true labels, Cols = predicted labels")
    print(cm)

    print("\nGLOBAL MULTICLASS METRICS")
    print(f"Accuracy            : {multiclass_metrics['accuracy']:.4f}")
    print(f"Precision (macro)   : {multiclass_metrics['precision_macro']:.4f}")
    print(f"Recall (macro)      : {multiclass_metrics['recall_macro']:.4f}")
    print(f"F1-score (macro)    : {multiclass_metrics['f1_macro']:.4f}")
    print(f"Precision (weighted): {multiclass_metrics['precision_weighted']:.4f}")
    print(f"Recall (weighted)   : {multiclass_metrics['recall_weighted']:.4f}")
    print(f"F1-score (weighted) : {multiclass_metrics['f1_weighted']:.4f}")

    print("\nPER-CLASS MULTICLASS METRICS")
    for i, lab in enumerate(labels):
        print(
            f"Class {lab}: "
            f"support={int(multiclass_metrics['support'][i])}, "
            f"precision={multiclass_metrics['per_class_precision'][i]:.4f}, "
            f"recall={multiclass_metrics['per_class_recall'][i]:.4f}, "
            f"f1={multiclass_metrics['per_class_f1'][i]:.4f}"
        )


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

def read_df_annot_from_json(json_path: Union[Path, str]) -> Optional[pd.DataFrame]:
    """
    Read and process annotations from a JSON file.

    Supported JSON formats
    ----------------------
    Old format:
        {
            ...
            "rpeaks": [
                {"sampleIndex": ..., "annotationValue": ...},
                ...
            ]
        }

    New format:
        {
            ...
            "rpeaks": {
                "__info": "...",
                "human": [...],
                "ml": [...],
                "merged": [...]
            }
        }

    For the new format, this function uses:
        rpeaks["merged"]

    Parameters
    ----------
    json_path : Path | str
        Full path to the JSON annotation file.

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns ['rpeak', 'annot'],
        or None if file does not exist.

    Raises
    ------
    ValueError
        If JSON structure is invalid or required fields are missing.
    TypeError
        If input data types are invalid.
    RuntimeError
        If JSON cannot be parsed or normalized.
    """
    all_beats = {'N': 0, 'S': 1, 'V': 2, 'U': 3}

    file_path = Path(json_path).expanduser().resolve()

    if not file_path.exists():
        return None

    if not file_path.is_file():
        raise ValueError(f"JSON path exists but is not a file: {file_path}")

    try:
        with open(file_path, "r", encoding="UTF-8", errors="ignore") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file: {file_path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read JSON file: {file_path}") from exc

    if not isinstance(data, dict):
        raise TypeError(
            f"Top-level JSON object must be dict, got {type(data).__name__} in file {file_path}"
        )

    if "rpeaks" not in data:
        raise ValueError(f"Key 'rpeaks' not found in JSON file: {file_path}")

    rpeaks_raw = data["rpeaks"]

    if rpeaks_raw is None:
        raise ValueError(f"Key 'rpeaks' is None in JSON file: {file_path}")

    # ------------------------------------------------------------------
    # Support both formats:
    # 1) old: rpeaks is a list
    # 2) new: rpeaks is a dict with keys: human, ml, merged
    # ------------------------------------------------------------------
    if isinstance(rpeaks_raw, list):
        rpeaks = rpeaks_raw

    elif isinstance(rpeaks_raw, dict):
        if "merged" not in rpeaks_raw:
            raise ValueError(
                f"New-format JSON detected, but key 'rpeaks[\"merged\"]' not found in file: {file_path}"
            )

        rpeaks = rpeaks_raw["merged"]

        if rpeaks is None:
            raise ValueError(
                f"Key 'rpeaks[\"merged\"]' is None in JSON file: {file_path}"
            )

        if not isinstance(rpeaks, list):
            raise TypeError(
                f"Key 'rpeaks[\"merged\"]' must be a list, got {type(rpeaks).__name__} in file {file_path}"
            )

    else:
        raise TypeError(
            f"Key 'rpeaks' must be either list (old format) or dict (new format), "
            f"got {type(rpeaks_raw).__name__} in file {file_path}"
        )

    if len(rpeaks) == 0:
        return pd.DataFrame(columns=["rpeak", "annot"])

    try:
        norm = pd.json_normalize(rpeaks)
        df_norm = cast(pd.DataFrame, norm)
    except Exception as exc:
        raise RuntimeError(f"pd.json_normalize failed for file: {file_path}") from exc

    if df_norm.empty:
        return pd.DataFrame(columns=["rpeak", "annot"])

    required_cols = ["sampleIndex", "annotationValue"]
    missing_cols = [col for col in required_cols if col not in df_norm.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns {missing_cols} in normalized rpeaks data from file: {file_path}"
        )

    try:
        rpeak_series = pd.to_numeric(df_norm["sampleIndex"], errors="raise").astype(int)
    except Exception as exc:
        raise ValueError(
            f"Column 'sampleIndex' contains non-numeric or invalid values in file: {file_path}"
        ) from exc

    try:
        annot_series = (
            df_norm["annotationValue"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map(all_beats)
            .fillna(0)
            .astype(int)
        )
    except Exception as exc:
        raise ValueError(
            f"Failed to map 'annotationValue' to numeric labels in file: {file_path}"
        ) from exc

    df_annot = pd.DataFrame({
        "rpeak": rpeak_series,
        "annot": annot_series,
    })

    return df_annot


# ======================================================================================
#                                        MAIN
# ======================================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate accuracy of ectopic beat detection on ZIVE data"
    )
    ap.add_argument(
        "--dir",
        type=Path,
        required=True,
        help="Directory containing ECG .npy files and corresponding JSON metadata",
    )
    ap.add_argument(
        "--exclude-list",
        type=Path,
        help="File containing list of ECG basenames to exclude (one basename per line, without extensions)",
    )
    ap.add_argument(
        "--cfg-denoising",
        type=Path,
        required=True,
        help="Denoising config path",
    )
    ap.add_argument(
        "--unet-model-dir",
        type=Path,
        required=True,
        help="Model directory for denoising/motions",
    )
    ap.add_argument(
        "--cfg-ectopy",
        type=Path,
        required=True,
        help="Ectopy config path",
    )
    ap.add_argument(
        "--ectopy-model-dir",
        type=Path,
        required=True,
        help="Model directory for ectopy detection",
    )
    ap.add_argument(
        "--fs",
        type=int,
        default=200,
        help="Sampling frequency (Hz). Default: 200",
    )
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="Silence stdout during denoising/stats",
    )
    ap.add_argument(
        "--disable-motions",
        action="store_true",
        help="Disable motions stage in denoising pipeline",
    )
    ap.add_argument(
        "--denoising",
        action="store_true",
        help="Enable denoising pipeline",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=20,
        help="Tolerance in samples for matching predicted and annotated R-peaks. Default: 20",
    )
    ap.add_argument(
        "--keep-u-class",
        action="store_true",
        help="Keep annot == 3 (U class) in evaluation. By default it is removed.",
    )
    ap.add_argument(
        "--all-records",
        action="store_true",
        help="Process all records. By default only first 5 records are processed.",
    )
    ap.add_argument(
        "--global-binary-metrics",
        action="store_true",
        help="Print global binary confusion matrix and binary metrics.",
    )

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

    print(f"total_json       : {summary.total_json}")
    print(f"excluded         : {summary.excluded}")
    print(f"matched          : {summary.matched}")
    print(f"unmatched_json   : {summary.unmatched_json}")
    print(f"records returned : {len(records)}")

    #               ++++++++++++++++++++++++ PIPELINE PARENGIMAS

    bundle = prepare_pipelines(args)
    agg = AggregateMetrics()

    #               ++++++++++++++++++++++++ TRIUKŠMŲ VALYMAS IR EKTOPINIŲ DŪŽIŲ DETEKTAVIMAS

    records_to_process = records if args.all_records else records[:5]

    print(f"\nFound {len(records)} matched records")
    print(f"Processing {len(records_to_process)} record(s)")

    for rec in records_to_process:
        print("\n" + "-" * 90)

        if rec.ecg_path is not None:
            print(rec.basename, rec.ecg_path.name, rec.json_path.name)
        else:
            print(rec.basename, "<missing ecg>", rec.json_path.name)

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

            # Pritaikykite pagal tai, ko tikisi jūsų read_df_annot(...)
            # Jei reikia basename be plėtinio, keiskite į rec.basename
            file_name = rec.json_path.name
            data_dir = rec.ecg_path.parent

            eval_res = evaluate_ectopy_classification_against_annotations(
                res_denoising=res_denoising,
                res_ectopy=res_ectopy,
                json_path=rec.json_path,
                fs=200,
                tolerance=20,
                drop_annot_u=True,
                verbose=True,
            )
            
            df_matched = eval_res["df_matched"]
            df_unmatched = eval_res["df_unmatched"]

            update_aggregate_metrics(agg, eval_res)

            _ = metadata, noise_stats, ectopy_stats, df_matched, df_unmatched
#             metadata, noise_stats, ectopy_stats, df_matched, df_unmatched jau turi kažkokias reikšmes
#             jos sudedamos į vieną tuple, tuple priskiriamas _
#           _ yra naudojamas, kad būtų aišku, jog šios reikšmės yra "naudojamos" (pvz. gali būti naudingos debugui ar tolimesnei analizei), bet šiuo metu nėra tiesiogiai naudojamos tolesniame kode. Tai gali būti naudinga, jei norite išlaikyti šias reikšmes atmintyje ar lengvai pasiekiamas, bet nenorite, kad jos trukdytų tolimesniam kodui, kuris šiuo metu yra fokusas.

        except Exception as exc:
            agg.n_records_failed_eval += 1
            print(f"WARN: failed processing for {rec.ecg_path.name}: {exc}")
            continue

    print_aggregate_report(
        agg,
        global_binary_metrics=args.global_binary_metrics,
    )


if __name__ == "__main__":
    main()