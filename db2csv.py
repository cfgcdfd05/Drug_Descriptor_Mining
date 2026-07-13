"""
drugbank_to_cid.py
------------------
Reads Drugbank.csv, adds a CID column (inserted after DRUGBANK_ID),
and updates the CSV in-place every 100 resolved rows.
Resumes automatically if interrupted — already-filled CID values are skipped.
"""

import csv
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

PUBCHEM_URL  = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/cids/JSON"
RETRIES      = 4
WAIT         = 6
WORKERS      = 7
FLUSH_EVERY  = 100

HERE     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "Drugbank.csv")


def fetch_cid(drugbank_id: str) -> int:
    url = PUBCHEM_URL.format(requests.utils.quote(drugbank_id))
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return r.json()["IdentifierList"]["CID"][0]
            if r.status_code == 404:
                print(f"  [MISS]  {drugbank_id} — not found.")
                return 0
            print(f"  [HTTP {r.status_code}] {drugbank_id} — attempt {attempt}/{RETRIES}")
        except Exception as e:
            print(f"  [ERR]   {drugbank_id} — attempt {attempt}/{RETRIES}: {e.__class__.__name__}")
        if attempt < RETRIES:
            time.sleep(WAIT)
    print(f"  [FAIL]  {drugbank_id} — giving up. CID=0.")
    return 0


def flush(rows: list, fieldnames: list) -> None:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process() -> None:
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader     = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        if "DRUGBANK_ID" not in fieldnames:
            raise ValueError("Column 'DRUGBANK_ID' not found in Drugbank.csv.")

        # Add CID column after DRUGBANK_ID if not already present
        if "CID" not in fieldnames:
            db_idx     = fieldnames.index("DRUGBANK_ID")
            fieldnames = fieldnames[:db_idx + 1] + ["CID"] + fieldnames[db_idx + 1:]

        rows = list(reader)
        for row in rows:
            row.setdefault("CID", "")

    total   = len(rows)
    pending = [i for i, r in enumerate(rows) if not str(r.get("CID", "")).strip()]

    print(f"Total rows   : {total}")
    print(f"Already done : {total - len(pending)}")
    print(f"Remaining    : {len(pending)}")
    print(f"Workers      : {WORKERS}\n")

    if not pending:
        print("Nothing to do — all rows already have a CID.")
        return

    lock            = Lock()
    already_done    = total - len(pending)
    completed_count = [0]

    def resolve(idx: int):
        db_id = rows[idx]["DRUGBANK_ID"].strip()
        cid   = fetch_cid(db_id) if db_id else 0

        with lock:
            rows[idx]["CID"] = cid
            completed_count[0] += 1
            done_session = completed_count[0]
            done_total   = already_done + done_session
            print(f"  [{done_total}/{total}] {db_id} → CID {cid if cid else 'MISS'}")
            if done_session % FLUSH_EVERY == 0:
                flush(rows, fieldnames)
                print(f"  [FLUSH] CSV updated ({done_total}/{total} rows done).")

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(resolve, i): i for i in pending}
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Flushing current progress to CSV...")
            flush(rows, fieldnames)
            print("[SAVED] CSV updated. Re-run to resume.")
            return

    # Final flush for the tail rows
    flush(rows, fieldnames)

    resolved = sum(1 for r in rows if str(r.get("CID", "")).strip() not in ("", "0"))
    print(f"\nDone. Drugbank.csv updated.") 
    print(f"Resolved: {resolved}/{total}  |  Failed/missing: {total - resolved}/{total}")


if __name__ == "__main__":
    process()