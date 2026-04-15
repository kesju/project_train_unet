

from pathlib import Path


# 3 galimi darbo režimai:
# 1) --denoising
# 2) --denoising + --disable-motions
# 3) be --denoising

from ecg_denoising_pipeline import DenoisingPipelineConfig, ECGDenoisingPipeline, resolve_model_path
from ecg_ectopy_pipeline import ECGEctopyPipeline


def prepare_denoising_pipeline_cfg(
    denoising_config_path: Path,
    denoising_model_dir: Path,
    denoising: bool = True,
    disable_motions: bool = False,
    
) -> DenoisingPipelineConfig:
    if not denoising_config_path.exists():
        raise FileNotFoundError(f"Config file not found: {denoising_config_path}")

    cfg = load_denoising_config_yaml(str(denoising_config_path))
    if check_denoising_config is not None:
        check_denoising_config(cfg)

    cfg.motions.model_name = resolve_model_path(denoising_model_dir, cfg.motions.model_name)
    cfg.filter.enabled = True

    if denoising and not disable_motions:
        print("CASE 1: denoising pipeline is ENABLED, including motions stage.")
        if not denoising_model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {denoising_model_dir}")
        
        cfg.outliers.enabled = True
        cfg.rdropouts.enabled = True
        cfg.motions.enabled = True
            
    elif denoising and disable_motions:
        print("CASE 2: denoising pipeline is ENABLED, but motions stage is DISABLED.")
        
        if not denoising_model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {denoising_model_dir}")
        
        cfg.outliers.enabled = True
        cfg.rdropouts.enabled = True
        cfg.motions.enabled = False
        
    else:
        print("CASE 3: denoising pipeline is DISABLED.")
        cfg.outliers.enabled = False
        cfg.rdropouts.enabled = False
        cfg.motions.enabled = False

    return cfg


def prepare_ectopy_pipeline_cfg(
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



cfg_denoising_path = args.cfg_denoising
denoising_model_dir = args.unet_model_dir
print(f"\nDenoising config path: {cfg_denoising_path}")
print(f"Denoising model directory: {denoising_model_dir}")

if args.denoising and not args.disable_motions:
    print("CASE 1: denoising pipeline is ENABLED, including motions stage.")
 #  Preparing the Denoising pipeline with configuration and model paths
    print("\n*****Denoising pipeline config:")
    
    cfg_denoising = prepare_denoising_pipeline_cfg(cfg_denoising_path, denoising_model_dir)
    MARKER = create_unet_marker(cfg_denoising)
    print(f"\nUNet marker: {MARKER}\n")
    denoising_pipe = ECGDenoisingPipeline(cfg_denoising)
    
elif args.denoising and args.disable_motions:
    print("CASE 2: denoising pipeline is ENABLED, but motions stage is DISABLED.")

    cfg_denoising_disabled_motions = prepare_denoising_pipeline_cfg(
    denoising_config_path=cfg_denoising_path,
    denoising_model_dir=denoising_model_dir,
    denoising=True,
    disable_motions=True,
    )
    denoising_pipe_disabled_motions = ECGDenoisingPipeline(cfg_denoising_disabled_motions)
else:
    
    print("CASE 3: denoising pipeline is DISABLED.")
    print("\nDenoising is disabled. Existing Excel values in out/rdr/mra/tp_pct and ectN/ectS/ectV/ectU will be kept unchanged.")
    MARKER = ""
    noise_stats = {}
    denoising_model_dir = "Not used"
    cfg_denoising = "Not used"
    cfg_ectopy = "Not used"
    ectopy_pipe = None
    denoising_pipe = None
    cfg_denoising_disabled_denoising = prepare_denoising_pipeline_cfg(
    denoising_config_path=cfg_denoising_path,
    denoising_model_dir=None,
    denoising=False,
    disable_motions=True,
    )
    denoising_pipe_disabled_denoising = ECGDenoisingPipeline(cfg_denoising_disabled_denoising)

    print("\nDenoising will be performed for each record using the specified config and model.")

    # prepare the denoising pipeline (if needed) ++++++++++++++++++++++++++++++++++++++
    
     
    # PREPARE ECTOPY DETECTION PART 
    
    cfg_ectopy = prepare_ectopy_pipeline_cfg(
        ectopy_config_path=args.cfg_ectopy,
        ectopy_model_dir=args.ectopy_model_dir
    )
    ectopy_pipe = ECGEctopyPipeline(cfg_ectopy)



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
    
    
    # 3 galimi darbo režimai:
    # 1) --denoising
    # 2) --denoising + --disable-motions
    # 3) be --denoising

    if args.denoising and not args.disable_motions:
        print("CASE 1: denoising pipeline is ENABLED, including motions stage.")
        x = load_ecg_npy(rec.ecg_path)
        res_denoising = None
        # denoising and getting noise stats
        if denoising_pipe is not None:
                res_denoising = denoising_pipe.run(x, gaps_indices=[])
        noise_stats = calc_noise_stats_from_denoised_result(res_denoising) or {}
        print(
            f"Noise stats for {rec.ecg_path.name}: "
            f"out={noise_stats['out']}, rdr={noise_stats['rdr']}, "
            f"mra={noise_stats['mra']}, tp_pct={noise_stats['tp_pct']:.1f} "
            f"with enabled detecting motions"
        )
        print()
        res_ectopy = ectopy_pipe.run(res_denoising, fs=args.fs)
        ectopy_stats = compute_ectopy_stats(res_ectopy)
        print(f"ectopy_stats={ectopy_stats} with enabled detecting motions")
        
    elif args.denoising and args.disable_motions:
        print("CASE 2: denoising pipeline is ENABLED, but motions stage is DISABLED.")
        x = load_ecg_npy(rec.ecg_path)
        res_denoising_disabled_motions = None
        # denoising and getting noise stats
        if denoising_pipe_disabled_motions is not None:
            res_denoising_disabled_motions = denoising_pipe_disabled_motions.run(x, gaps_indices=[])
        noise_stats_disabled_motions = calc_noise_stats_from_denoised_result(
            res_denoising_disabled_motions) or {}
        print(
            f"Noise stats for {rec.ecg_path.name}: "
            f"out={noise_stats_disabled_motions['out']}, "
            f"rdr={noise_stats_disabled_motions['rdr']}, "
            f"mra={noise_stats_disabled_motions['mra']}, "
            f"tp_pct={noise_stats_disabled_motions['tp_pct']:.1f} "
            f"with disabled detecting motions"
        )
        res_ectopy = ectopy_pipe.run(res_denoising_disabled_motions, fs=args.fs)
        ectopy_stats = compute_ectopy_stats(res_ectopy)
        print(f"ectopy_stats={ectopy_stats} with disabled detecting motions")
    
    else:
        print("CASE 3: denoising pipeline is DISABLED.")
        x = load_ecg_npy(rec.ecg_path)
        res_denoising_disabled_denoising = denoising_pipe_disabled_denoising.run(x, gaps_indices=[])
        res_ectopy = ectopy_pipe.run(res_denoising_disabled_denoising, fs=args.fs)
        ectopy_stats = compute_ectopy_stats(res_ectopy)
        print(f"ectopy_stats={ectopy_stats} with disabled denoising")
