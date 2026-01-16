from pathlib import Path


def find_project_root_by_name(target="S-ITP-25-9", start: Path | str | None = None) -> Path:
    """Walk up from `start` (or CWD) until a directory named `target` is found."""
    start_path = Path(start or Path.cwd()).resolve()
    for p in (start_path, *start_path.parents):
        if p.name == target:
            return p
    raise FileNotFoundError(f"Could not find '{target}' above {start_path}")

