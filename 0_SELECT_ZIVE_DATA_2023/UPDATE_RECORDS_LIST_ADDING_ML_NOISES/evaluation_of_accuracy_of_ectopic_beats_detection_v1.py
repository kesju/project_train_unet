from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import argparse

# naudosime funkcijas iš zive_data_read_utils.py
from zive_data_read_utils import load_ecg_npy, list_ecg_records, read_json_file

try:
    from ecg_denoising_pipeline import (
        ECGDenoisingPipeline,
        load_denoising_config_yaml,
        DenoisingPipelineConfig,
        resolve_model_path,
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
    )
except Exception as exc:
    raise ImportError(
        "Cannot import ECG ectopy pipeline helpers. "
        "Update imports to match your project structure.\n"
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

    # Reikalavimas: bent dvi eilutės kiekvienam atvejui:
    # - print, kuris parodo CASE
    # - x = load_ecg_npy(rec.ecg_path)
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

    res_ectopy = bundle.ectopy_pipe.run(res_denoising, fs=fs)
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

    # Naudosime šiuos signalus ir metadata toliau, kad įvertintume ektoopinių dūžių aptikimo tikslumą.
    # Galime naudoti metadata, kad sužinotume, kur yra ektoopiniai dūžiai, o signalą - kad patikrintume,
    # ar mūsų aptikimo algoritmas juos teisingai identifikuoja.

    #               ++++++++++++++++++++++++ PIPELINE PARENGIMAS

    bundle = prepare_pipelines(args)

    #               ++++++++++++++++++++++++ TRIUKŠMŲ VALYMAS IR EKTOPINIŲ DŪŽIŲ DETEKTAVIMAS

    print(f"\nFound {len(records)} matched records")
    for rec in records[:5]:
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

            # Čia toliau galima lyginti metadata anotacijas su ectopy rezultatais
            # ir skaičiuoti accuracy / sensitivity / specificity / confusion matrix ir pan.

            # laikome metadata panaudotą, kad būtų aišku jog ji tikrai nuskaityta
            _ = metadata, noise_stats, ectopy_stats

        except Exception as exc:
            print(f"WARN: failed processing for {rec.ecg_path.name}: {exc}")
            continue

    # Toliau galima pridėti:
    # - rpeak perskaičiavimą į pradinį signalą
    # - merge su gydytojo anotacijomis
    # - classification metrics skaičiavimą


if __name__ == "__main__":
    main()