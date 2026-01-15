#!/usr/bin/env python3

"""
python3 set_noises_empty.py             # run in current folder
python3 set_noises_empty.py /path/to/folder
Notes:

Works only in the given folder (non-recursive), as you asked.

Touches every "noises" key anywhere in the JSON (top-level or nested).

Creates a yourfile.json.bak backup before overwriting. If you’re happy after checking, you can delete the .bak files.
"""


import json
from pathlib import Path
import shutil
import sys

def replace_noises(obj) -> bool:
    """
    Recursively set any 'noises' key to [].
    Returns True if any change was made.
    """
    changed = False
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            if k == "noises":
                if obj[k] != []:
                    obj[k] = []
                    changed = True
            # Recurse into values
            sub = obj[k]
            if isinstance(sub, (dict, list)):
                if replace_noises(sub):
                    changed = True
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, (dict, list)):
                if replace_noises(item):
                    changed = True
    return changed

def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    if not folder.is_dir():
        print(f"Not a directory: {folder}")
        sys.exit(1)

    json_files = sorted(folder.glob("*.json"))
    if not json_files:
        print("No .json files found.")
        return

    modified, skipped, failed = 0, 0, 0
    for jf in json_files:
        try:
            text = jf.read_text(encoding="utf-8")
            data = json.loads(text)
        except Exception as e:
            print(f"[ERROR] {jf.name}: failed to read/parse JSON: {e}")
            failed += 1
            continue

        changed = replace_noises(data)

        if changed:
            # Make a backup once per file
            backup = jf.with_suffix(jf.suffix + ".bak")
            try:
                shutil.copy2(jf, backup)
            except Exception as e:
                print(f"[WARN] {jf.name}: could not write backup: {e}")

            try:
                with jf.open("w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                    f.write("\n")
                print(f"[MODIFIED] {jf.name}")
                modified += 1
            except Exception as e:
                print(f"[ERROR] {jf.name}: failed to write file: {e}")
                failed += 1
        else:
            print(f"[SKIPPED] {jf.name} (no 'noises' changes needed)")
            skipped += 1

    print(f"\nDone. Modified: {modified}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main()
