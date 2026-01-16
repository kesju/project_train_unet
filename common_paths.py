from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SUBPROJECT_NAMES = ("1_PREPARE_TRAIN_UNET_DATA", "4_TEST_UNET", "2_TRAIN_UNET")


def _find_repo_root() -> Path:
    """
    Resolve repo root.

    Priority:
      1) REPO_ROOT env var, if set.
      2) Walk upwards from script dir (.py) or cwd (.ipynb) until we find:
           - a folder named PROJECT_TRAIN_UNET, OR
           - a marker file/folder: .git or pyproject.toml
      3) Fallback to start directory.
    """
    v = os.environ.get("REPO_ROOT")
    if v:
        return Path(v).expanduser().resolve()

    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()

    markers = (".git", "pyproject.toml", "requirements.txt")
    for p in (start, *start.parents):
        if p.name == "PROJECT_TRAIN_UNET":
            return p
        if any((p / m).exists() for m in markers):
            # If markers are found in a parent that is not named PROJECT_TRAIN_UNET,
            # we still accept it as repo root (useful if you rename later).
            return p

    return start


def _infer_subproject(repo_root: Path) -> str:
    """
    Determine active subproject.

    Priority:
      1) SUBPROJECT env var, if set (must be one of SUBPROJECT_NAMES).
      2) Infer from script dir (.py) or cwd (.ipynb):
         if the start path is inside repo_root/<subproject>/..., pick that subproject.
      3) Fallback to 2_TRAIN_UNET.
    """
    v = os.environ.get("SUBPROJECT")
    if v:
        if v not in SUBPROJECT_NAMES:
            raise ValueError(
                f"SUBPROJECT={v!r} is not valid. Allowed: {SUBPROJECT_NAMES}"
            )
        return v

    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd().resolve()

    for name in SUBPROJECT_NAMES:
        sp_root = (repo_root / name).resolve()
        try:
            start.relative_to(sp_root)
            return name
        except Exception:
            pass

    return "2_TRAIN_UNET"


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    subproject: str
    subproject_root: Path
    data_dir: Path
    results_dir: Path
    models_dir: Path
    logs_dir: Path

    def mkdirs(self) -> None:
        # Shared + per-subproject
        for d in (self.data_dir, self.results_dir, self.models_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)


def build_paths() -> Paths:
    repo_root = _find_repo_root()
    subproject = _infer_subproject(repo_root)
    subproject_root = (repo_root / subproject).resolve()

    if not subproject_root.exists():
        raise FileNotFoundError(f"Subproject folder not found: {subproject_root}")

    p = Paths(
        repo_root=repo_root,
        subproject=subproject,
        subproject_root=subproject_root,
        data_dir=repo_root / "data",                  # shared
        results_dir=subproject_root / "results",      # local
        models_dir=subproject_root / "models",        # local
        logs_dir=subproject_root / "logs",            # local
    )
    p.mkdirs()
    return p


PATHS = build_paths()

# Optional: quick visibility while developing
if __name__ == "__main__":
    print("REPO_ROOT       :", PATHS.repo_root)
    print("SUBPROJECT      :", PATHS.subproject)
    print("SUBPROJECT_ROOT :", PATHS.subproject_root)
    print("DATA_DIR        :", PATHS.data_dir)
    print("RESULTS_DIR     :", PATHS.results_dir)
    print("MODELS_DIR      :", PATHS.models_dir)
    print("LOGS_DIR        :", PATHS.logs_dir)
