import re
import pandas as pd
import numpy as np

DB_PATH  = "Drugbank.csv"
GEN_PATH = "Drugbank_Generated.csv"

# Columns to compare — same name in both files after normalise_columns.py
COMPARE_COLS = [
    "Number of Atoms",
    "Molecular Formula",
    "Molecular Composition",
    "Molecular Weight",
    "Exact Mol. Weight",
    "Net Formal Charge",
    "Num_H_Acceptors_Lipinski",
    "Num_H_Donors_Lipinski",
    "ALogP",
    "Num_RotatableBonds",
    "Molecular_PolarSurfaceArea",
    "Num_H_Acceptors",
    "Num_H_Donors",
    "COMMON_NAME",
    "CAS_NUMBER",
    "UNII",
    "DRUGBANK_ID",
]

# ── Normalisation ─────────────────────────────────────────────────────────────
def norm_formula(s): return re.sub(r'\s+', '', s)
def norm_name(s):
    s = s.strip().lower()
    return re.sub(r'^(l-|d-|dl-)', '', s)

def parse_composition(s):
    result = {}
    for part in s.split(','):
        m = re.match(r'\s*([A-Z][a-z]?):\s*([\d.]+)', part)
        if m:
            result[m.group(1)] = round(float(m.group(2)), 3)
    return result

def composition_match(g_str, d_str):
    try:
        g, d = parse_composition(str(g_str)), parse_composition(str(d_str))
        return set(g) == set(d) and all(abs(g[k] - d[k]) < 0.001 for k in g)
    except Exception:
        return False

# ── Comparisons ───────────────────────────────────────────────────────────────
def compare_string(col, g_arr, d_arr):
    if col == 'Molecular Composition':
        exact = sum(composition_match(gv, dv) for gv, dv in zip(g_arr, d_arr))
    else:
        def norm(arr):
            s = pd.Series(arr).fillna('').astype(str).str.strip().str.lower()
            if col == 'Molecular Formula': s = s.apply(norm_formula)
            if col == 'COMMON_NAME':       s = s.apply(norm_name)
            return s.to_numpy()
        exact = (norm(g_arr) == norm(d_arr)).sum()
    return {'type': 'string', 'exact_pct': round(100 * exact / len(g_arr), 1)}

def compare_numeric(g_arr, d_arr):
    g = pd.to_numeric(pd.Series(g_arr), errors='coerce').to_numpy()
    d = pd.to_numeric(pd.Series(d_arr), errors='coerce').to_numpy()
    mask = ~(np.isnan(g) | np.isnan(d))
    g, d = g[mask], d[mask]
    n = len(g)
    if n == 0:
        return {'type': 'numeric', 'n': 0, 'bias': None, 'median': None,
                'mae': None, 'std': None, 'outliers': None}
    err = d - g    # actual - generated, raw units
    std = float(np.std(err))
    outliers = int(np.sum(np.abs(err) > 2 * std)) if std > 0 else 0
    return {
        'type':     'numeric',
        'n':        n,
        'bias':     round(float(np.mean(err)), 3),
        'median':   round(float(np.median(err)), 3),
        'mae':      round(float(np.mean(np.abs(err))), 3),
        'std':      round(std, 3),
        'outliers': outliers,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    db  = pd.read_csv(DB_PATH,  on_bad_lines='skip')
    gen = pd.read_csv(GEN_PATH, on_bad_lines='skip')

    # Drop duplicate CIDs — keep first occurrence
    db  = db.drop_duplicates(subset='CID', keep='first')
    gen = gen.drop_duplicates(subset='CID', keep='first')

    # Align on common CIDs
    common = set(db['CID']) & set(gen['CID'])
    db  = db[db['CID'].isin(common)].sort_values('CID').reset_index(drop=True)
    gen = gen[gen['CID'].isin(common)].sort_values('CID').reset_index(drop=True)

    print(f"Matched rows: {len(db)}\n")
    assert len(db) == len(gen), "Row count mismatch after alignment"

    results = {}
    for col in COMPARE_COLS:
        if col not in gen.columns or col not in db.columns:
            print(f"[skip] '{col}' missing from one or both files")
            continue
        g_arr = gen[col].to_numpy()
        d_arr = db[col].to_numpy()
        is_num = (pd.api.types.is_numeric_dtype(gen[col]) and
                  pd.api.types.is_numeric_dtype(db[col]))
        results[col] = (compare_numeric(g_arr, d_arr)
                        if is_num else
                        compare_string(col, g_arr, d_arr))

    # ── Print numeric ─────────────────────────────────────────────────────────
    num_cols = {k: v for k, v in results.items() if v['type'] == 'numeric'}
    print(f"  {'Column':<30} {'N':>6}  {'Bias':>12}  {'Median':>12}  {'MAE':>12}  {'Std':>12}  {'Outliers(>2σ)':>14}")
    print("-" * 100)
    for col, st in num_cols.items():
        n_s      = f"{st['n']:>6}"           if st['n']        is not None else "   N/A"
        bias_s   = f"{st['bias']:>+12.3f}"   if st['bias']     is not None else "         N/A"
        median_s = f"{st['median']:>+12.3f}" if st['median']   is not None else "         N/A"
        mae_s    = f"{st['mae']:>12.3f}"     if st['mae']      is not None else "         N/A"
        std_s    = f"{st['std']:>12.3f}"     if st['std']      is not None else "         N/A"
        out_s    = f"{st['outliers']:>14}"   if st['outliers'] is not None else "           N/A"
        print(f"  {col:<30} {n_s}  {bias_s}  {median_s}  {mae_s}  {std_s}  {out_s}")

    print()
    print("  err = actual - generated  (raw units)")
    print("  Bias/Median +ve → under-predict  |  -ve → over-predict  |  Outliers: |err| > 2σ")

if __name__ == '__main__':
    run()