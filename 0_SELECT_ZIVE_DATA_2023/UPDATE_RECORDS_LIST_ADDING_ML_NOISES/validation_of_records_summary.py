"""
In the records summary Excel file, filename has the form userPrefix_recordingId, for example 1000_1.
The first part (1000) represents the user-level identifier, and the second part (1) 
represents the recording-level identifier.
The validation rule is that all rows with the same filename prefix 
before _ must have the same userId, and different filename prefixes must 
correspond to different userId values.
Therefore, the mapping between filename prefix and userId must be one-to-one across the whole table.

https://chatgpt.com/c/69bbab8b-be1c-8395-90a9-831b8d4c6158

python validation_of_records_summary.py

"""
from pathlib import Path
import pandas as pd


EXCEL_FILE = Path("/home/kesju/DI/2025_ZIVEO/PROJECT_TRAIN_UNET/0_SELECT_ZIVE_DATA_2023/AtsisiuntimasZiveDuomenu/Atsisiusti_visi_anotuoti_duomenys/visi_zive_irasai_annot-Darb.xlsx")
SHEET_NAME = "Records"




def main() -> None:
    # Read Excel
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

    # Normalize column names (remove accidental leading/trailing spaces)
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = ["filename", "userId"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only rows where both filename and userId are present
    work = df[["filename", "userId"]].copy()
    work = work.dropna(subset=["filename", "userId"])

    # Convert to string and strip spaces
    work["filename"] = work["filename"].astype(str).str.strip()
    work["userId"] = work["userId"].astype(str).str.strip()

    # Extract prefix before "_"
    work["file_prefix"] = work["filename"].str.split("_").str[0]

    # Optional: detect malformed filenames (without "_")
    malformed = work[~work["filename"].str.contains("_", regex=False)].copy()

    # ----------------------------------------------------------
    # Check 1:
    # each file_prefix must map to exactly one userId
    # ----------------------------------------------------------
    prefix_to_user_counts = (
        work.groupby("file_prefix")["userId"]
        .nunique()
        .reset_index(name="userId_count")
    )

    bad_prefixes = prefix_to_user_counts[prefix_to_user_counts["userId_count"] > 1].copy()

    bad_prefix_rows = pd.DataFrame()
    if not bad_prefixes.empty:
        bad_prefix_rows = (
            work[work["file_prefix"].isin(bad_prefixes["file_prefix"])]
            .sort_values(["file_prefix", "userId", "filename"])
            .copy()
        )

    # ----------------------------------------------------------
    # Check 2:
    # each userId must map to exactly one file_prefix
    # ----------------------------------------------------------
    user_to_prefix_counts = (
        work.groupby("userId")["file_prefix"]
        .nunique()
        .reset_index(name="prefix_count")
    )

    bad_users = user_to_prefix_counts[user_to_prefix_counts["prefix_count"] > 1].copy()

    bad_user_rows = pd.DataFrame()
    if not bad_users.empty:
        bad_user_rows = (
            work[work["userId"].isin(bad_users["userId"])]
            .sort_values(["userId", "file_prefix", "filename"])
            .copy()
        )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("=" * 70)
    print(f"File: {EXCEL_FILE.name}")
    print(f"Sheet: {SHEET_NAME}")
    print(f"Checked rows: {len(work)}")
    print("=" * 70)

    if malformed.empty:
        print("Malformed filename rows (no '_'): 0")
    else:
        print(f"Malformed filename rows (no '_'): {len(malformed)}")
        print(malformed.head(20).to_string(index=False))
        print()

    print(f"Prefixes mapping to more than one userId: {len(bad_prefixes)}")
    print(f"userId values mapping to more than one prefix: {len(bad_users)}")
    print()

    if bad_prefixes.empty and bad_users.empty and malformed.empty:
        print("OK: the table satisfies the required one-to-one relationship.")
    else:
        print("FAIL: inconsistencies were found.")

        if not bad_prefixes.empty:
            print("\n--- Problem type 1: same filename prefix -> multiple userId values ---")
            print(bad_prefix_rows.to_string(index=False))

        if not bad_users.empty:
            print("\n--- Problem type 2: same userId -> multiple filename prefixes ---")
            print(bad_user_rows.to_string(index=False))

    # ----------------------------------------------------------
    # Save detailed report to Excel
    # ----------------------------------------------------------
    out_file = EXCEL_FILE.with_name(EXCEL_FILE.stem + "_prefix_userid_check.xlsx")

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        prefix_to_user_counts.to_excel(writer, sheet_name="prefix_to_user_summary", index=False)
        user_to_prefix_counts.to_excel(writer, sheet_name="user_to_prefix_summary", index=False)

        if not bad_prefix_rows.empty:
            bad_prefix_rows.to_excel(writer, sheet_name="bad_prefix_rows", index=False)

        if not bad_user_rows.empty:
            bad_user_rows.to_excel(writer, sheet_name="bad_user_rows", index=False)

        if not malformed.empty:
            malformed.to_excel(writer, sheet_name="malformed_filenames", index=False)

    print()
    print(f"Detailed report saved to: {out_file}")


if __name__ == "__main__":
    main()