"""
generate_drugbank.py
--------------------
Reads Drugbank.csv, takes the first 50 rows with a non-zero CID,
runs each through main.py's run_pipeline(), and writes results to
Drugbank_Generated.csv in the same directory.

List fields:
  secondary_accession_numbers → Secondary_Accession_1, Secondary_Accession_2
  synonyms                    → Synonym_1, Synonym_2, Synonym_3
  molecular_composition       → single cell e.g. "C: 0.6008, H: 0.0446, O: 0.3546"
"""

import csv
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as pipeline

HERE        = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV   = os.path.join(HERE, "Drugbank.csv")
OUTPUT_CSV  = os.path.join(HERE, "Drugbank_Generated.csv")
ERROR_LOG   = os.path.join(HERE, "Drugbank_Generated_errors.log")  # failed CIDs logged here, not as empty rows
MAX_ROWS    = 10000

FIELDNAMES = [
    "cid",
    "drugbank_id",
    "Secondary_Accession_1", "Secondary_Accession_2",
    "common_name",
    "cas_number",
    "unii",
    "Synonym_1", "Synonym_2", "Synonym_3",
    "smiles",
    "number_of_atoms",
    "net_formal_charge",
    "molecular_formula",
    "molecular_composition",
    "molecular_weight",
    "exact_mol_weight",
    "num_h_acceptors_lipinski",
    "num_h_donors_lipinski",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "alogp",
    "molecular_polar_surface_area",
]


def flatten(result: dict) -> dict:
    """Flatten a run_pipeline() result dict into a CSV row."""
    row = {}

    # Scalar fields — direct copy
    for key in (
        "cid", "drugbank_id", "common_name", "cas_number", "unii",
        "smiles", "number_of_atoms", "net_formal_charge", "molecular_formula",
        "molecular_weight", "exact_mol_weight", "num_h_acceptors_lipinski",
        "num_h_donors_lipinski", "num_rotatable_bonds", "num_h_acceptors",
        "num_h_donors", "alogp", "molecular_polar_surface_area",
    ):
        row[key] = result.get(key, "")

    # secondary_accession_numbers → 2 columns
    san = result.get("secondary_accession_numbers") or []
    row["Secondary_Accession_1"] = san[0] if len(san) > 0 else ""
    row["Secondary_Accession_2"] = san[1] if len(san) > 1 else ""

    # synonyms → 3 columns
    syn = result.get("synonyms") or []
    row["Synonym_1"] = syn[0] if len(syn) > 0 else ""
    row["Synonym_2"] = syn[1] if len(syn) > 1 else ""
    row["Synonym_3"] = syn[2] if len(syn) > 2 else ""

    # molecular_composition → single cell string
    comp = result.get("molecular_composition")
    if isinstance(comp, dict) and comp:
        row["molecular_composition"] = ", ".join(
            f"{k}: {round(v, 4)}" for k, v in comp.items()
        )
    else:
        row["molecular_composition"] = ""

    return row


def load_done_cids(path: str) -> set[int]:
    """Return set of CIDs already written to the output CSV."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()
        # Handle both 'cid' (pre-normalise) and 'CID' (post-normalise)
        cid_col = next((c for c in reader.fieldnames if c.upper() == "CID"), None)
        if cid_col is None:
            return set()
        for row in reader:
            try:
                cid = int(row[cid_col])
                if cid != 0:
                    done.add(cid)
            except (ValueError, TypeError):
                continue
    return done


def load_cids(path: str, skip: set[int]) -> list[int]:
    """Return first MAX_ROWS non-zero CIDs from Drugbank.csv, skipping already-done ones."""
    cids = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "CID" not in reader.fieldnames:
            raise ValueError("CID column not found in Drugbank.csv. Run drugbank_to_cid.py first.")
        for row in reader:
            try:
                cid = int(row["CID"])
            except (ValueError, TypeError):
                continue
            if cid != 0 and cid not in skip:
                cids.append(cid)
            if len(cids) >= MAX_ROWS:
                break
    return cids


def main():
    done_cids = load_done_cids(OUTPUT_CSV)
    if done_cids:
        print(f"[RESUME] {len(done_cids)} CIDs already in output — skipping them.")
    cids = load_cids(INPUT_CSV, skip=done_cids)
    total = len(cids)
    print(f"Found {total} non-zero CIDs to process (max {MAX_ROWS}).\n")

    failed: list[tuple[int, str]] = []

    # Append if resuming, write fresh if starting from scratch
    mode = "a" if done_cids else "w"
    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not done_cids:
            writer.writeheader()

        for i, cid in enumerate(cids, 1):
            print(f"[{i}/{total}] Running pipeline for CID {cid}...")
            try:
                result = pipeline.run_pipeline(str(cid), debug=False)
                row    = flatten(result)
                writer.writerow(row)
                f.flush()
                print(f"  → Written: {result.get('common_name', '?')} (CID {cid})\n")
            except Exception as e:
                # Do NOT write an empty row — it would drag down all compare.py metrics.
                # Log the failure and move on; resume will skip done CIDs automatically.
                msg = f"CID {cid}: {type(e).__name__}: {e}"
                print(f"  [ERROR] {msg} — skipping (logged to {os.path.basename(ERROR_LOG)}).\n")
                failed.append((cid, str(e)))

    # Write error log so failed CIDs are easy to inspect / retry
    if failed:
        with open(ERROR_LOG, "a", encoding="utf-8") as ef:
            for cid, err in failed:
                ef.write(f"{cid}\t{err}\n")
        print(f"\n{len(failed)} CID(s) failed — see '{ERROR_LOG}' for details.")

    print(f"Done. Results written to '{OUTPUT_CSV}'.")


if __name__ == "__main__":
    main()