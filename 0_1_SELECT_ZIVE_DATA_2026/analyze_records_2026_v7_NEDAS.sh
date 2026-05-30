# chmod +x analyze_records_2026_v7_NEDAS.sh

# NEDAS
  python analyze_records_2026_v7.py \
  --dir /Users/kesju/DI/DUOMENYS/DUOMENYS_NAUJI_2026/613b1d3a3d08d41d84cdc8f7 \
  --fs 200 \
  --cfg-denoising "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/denoising_config.yaml" \
  --unet-model-dir /Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_UNET \
  --disable-motions \
  --cfg-ectopy "/Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/CONFIG/ectopy_config.yaml" \
  --ectopy-model-dir /Users/kesju/DI/REPOS_ON_GITHUB_ARCH/project_train_unet/MODEL_VU_CNN \
  --out AtsisiustiDuomenys_2026/recordings-6072dcc694e6eac0851ab913-2026-05-29T12-38-20-370Z/records_summary.xlsx

