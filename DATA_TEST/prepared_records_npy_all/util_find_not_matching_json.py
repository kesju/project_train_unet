from pathlib import Path

folder = Path(".")  # change if needed

npys  = {p.stem.lower() for p in folder.glob("*.npy")}
jsons = {p.stem.lower() for p in folder.glob("*.json")}

json_without_npy = sorted(jsons - npys)
npy_without_json = sorted(npys - jsons)  # (optional) the opposite check

print("JSON without matching NPY:")
for name in json_without_npy:
    print(f"{name}.json")

# Optional: also show the reverse
print("\nNPY without matching JSON:")
for name in npy_without_json:
    print(f"{name}.npy")
