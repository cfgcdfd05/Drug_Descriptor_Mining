# Biomedical Property Reconstruction Pipeline

An automated three-tier pipeline that extracts, fuses, and verifies physicochemical and pharmacological descriptors for small molecules from scientific literature and public databases. Uses a **hybrid LLM → Code → LLM** architecture: broad LLM extraction, deterministic mathematical fusion, and LLM-based plausibility verification.

> **Note:** Runs fully local via [Ollama](https://ollama.com) using `deepseek-r1:70b` for both LLM layers. No API key or `.env` required. Tested on dual RTX 4090 (48GB VRAM).

---

## Architecture

```
[ Molecule Name / CID ]
         │
         ▼
  PubChem API — resolves CID; enriches with IUPAC + common name
         │
         ▼
  LAYER 1 — LLM1 (LLM1.py)
  Model: deepseek-r1:70b (Ollama, local), temperature=0.0
  • Emits every candidate value found — never resolves conflicts
  • Each leaf → list of {value, confidence (0–1), source_type}
  • Retries up to 3×; num_predict scales +1024 per retry on malformed JSON
         │
         ▼
  LAYER 2 — Deterministic Fusion (fusion.py)
  • Confidence threshold filtering (MIN_CONFIDENCE = 0.4)
  • Source-priority boosting: eff = conf × (1 + 0.1 × (priority − 1))
  • Per-field confidence capping (CONFIDENCE_CAP)
  • Plausibility floor/ceiling enforcement
  • Consensus hallucination detection (skipped for DETERMINISTIC_FIELDS)
  • Median-based outlier removal (absolute threshold or 5× ratio)
  • Type-aware fusion: float→weighted mean, int→weighted mode,
    categorical/smiles→highest confidence, list→dedup union,
    composition→per-element float fusion + renormalisation
         │
         ▼
  LAYER 3 — LLM2 (LLM2.py)
  Model: deepseek-r1:70b (Ollama, local), temperature=0.0
  • Verifies biological plausibility; fills missing string fields
  • Must NOT silently alter any fused numeric value
  • Retries up to 4× on malformed JSON; num_predict scales +1024 per retry (capped at 5120)
         │
         ▼
  main.py guard layer
  • Re-injects fused numerics (overrides any LLM2 drift)
  • String/list/composition dict guards
  • SMILES syntactic validation; clears on failure
  • Deterministic recomputation from molecular_formula (MW, exact mass, HAC, composition)
  • Formula↔MW, SMILES↔HAC, and composition-sum consistency checks
  • Integer type enforcement [INT-GUARD]
  • Null sentinels for unresolvable fields
  • Schema key audit; CID injection
         │
         ▼
  [ output.json ]
```

---

## Repository Structure

```
.
├── main.py               # Orchestrator and entry point
├── LLM1.py               # Layer 1 — multi-candidate extraction (Ollama, deepseek-r1:70b)
├── fusion.py             # Layer 2 — deterministic type-aware fusion engine
├── fusion_config.py      # All mathematical thresholds (no code changes needed)
├── LLM2.py               # Layer 3 — plausibility verification (Ollama, deepseek-r1:70b)
├── drug_descriptors.py   # Schema, field types, source priorities, deterministic field sets
├── Generate_drugbank.py  # Batch runner: processes Drugbank.csv → Drugbank_Generated.csv
├── db2csv.py             # Utility: resolves DrugBank IDs → PubChem CIDs, writes CID column to Drugbank.csv
├── compare.py            # Utility: compares Drugbank_Generated.csv against Drugbank.csv ground truth
├── Drugbank.csv          # Source data (DrugBank entries with resolved CIDs)
├── output.json           # Sample output from a completed run (git-ignored)
└── .gitignore
```

---

## Data Schema

Defined in `drug_descriptors.py` as `REQUIRED_SCHEMA`. Flat structure; default values encode the expected leaf type at runtime (`0` = int, `0.0` = float, `""` = str, `[]` = list, `{}` = dict).

```json
{
  "cid": 0,
  "drugbank_id": "",
  "secondary_accession_numbers": [],
  "common_name": "",
  "cas_number": "",
  "unii": "",
  "synonyms": [],
  "smiles": "",
  "number_of_atoms": 0,
  "net_formal_charge": 0,
  "molecular_formula": "",
  "molecular_composition": {},
  "molecular_weight": 0.0,
  "exact_mol_weight": 0.0,
  "num_h_acceptors_lipinski": 0,
  "num_h_donors_lipinski": 0,
  "num_rotatable_bonds": 0,
  "num_h_acceptors": 0,
  "num_h_donors": 0,
  "alogp": 0.0,
  "molecular_polar_surface_area": 0.0
}
```

| Field | Type | Notes |
|---|---|---|
| `cid` | int | PubChem CID — injected by `main.py`, never from LLM |
| `drugbank_id` | str | Primary DrugBank accession (e.g. `DB01050`) |
| `secondary_accession_numbers` | list[str] | Must not duplicate `drugbank_id` |
| `cas_number` | str | Canonical CAS of neutral free-acid/free-base form (lowest-numbered) |
| `smiles` | str | Canonical SMILES; cleared on validation failure |
| `number_of_atoms` | int | Non-hydrogen atom count; recomputed from formula when available |
| `molecular_composition` | dict | `{element: mass_fraction}`; fractions sum to 1.0 |
| `molecular_weight` | float | Average MW in Da |
| `exact_mol_weight` | float | Monoisotopic mass in Da; must differ from `molecular_weight` |
| `alogp` | float | Ghose–Crippen octanol-water partition coefficient |
| `molecular_polar_surface_area` | float | Topological PSA in Å² |

Adding a field to `drug_descriptors.py` automatically propagates through all three layers. The only other file that may need updating is `fusion_config.py` if the new field requires custom thresholds.

---

## Fusion Algorithm

`fusion.py` is entirely deterministic — zero LLM calls. Each candidate has the shape `{"value": ..., "confidence": 0–1, "source_type": "pubchem|chembl|drugbank|..."}`.

**Stage 1 — Source-Priority Boosting & Threshold**

```
eff = confidence × (1 + 0.1 × (priority − 1))
```

| Source | Priority | Multiplier |
|---|---|---|
| pubchem, drugbank | 5 | ×1.4 |
| chembl | 4 | ×1.3 |
| bindingdb | 3 | ×1.2 |
| literature | 2 | ×1.1 |
| other | 1 | ×1.0 |

Candidates with `eff < 0.4` are dropped permanently.

**Stage 2 — Per-Field Confidence Capping** — prevents LLMs from being overconfident on structural descriptors (e.g. `alogp` capped at 0.70, `molecular_weight` at 0.80).

**Stage 3 — Plausibility Window** — values outside configured floor/ceiling are discarded (e.g. MW must be 10–2000 Da). If all candidates fall outside the window, the field returns `None`.

**Stage 4 — Consensus Hallucination Detection** — if all numeric candidates in a non-deterministic field share exactly the same value, a `[FUSION] WARNING` is emitted and all confidences are capped to 0.5. Skipped for `DETERMINISTIC_FIELDS` (MW, formula, HAC, etc.) where identical values across sources are expected.

**Stage 5 — Outlier Removal** — absolute deviation from median for fields with known tolerances (MW: ±50 Da, exact mass: ±0.005 Da, alogp: ±0.5, PSA: ±10 Å²); 5× ratio rule for all others.

**Stage 6 — Type-Aware Fusion**

- `float` → confidence-weighted mean
- `integer` → weighted mode → cast to `int`; never produces `3.2857` for `num_rotatable_bonds`
- `categorical` → highest effective confidence; `cas_number` additionally prefers lexicographically smallest (earliest-registered) among tied candidates
- `smiles` → highest effective confidence string; validation deferred to `main.py`
- `list` → deduplicated union ordered by descending effective confidence

**Stage 7 — Composition Fusion** — `molecular_composition` is fused per element using float fusion, then renormalised so all fractions sum to exactly 1.0. Legacy string format (`"C: 0.600, H: 0.045"`) is parsed automatically.

---

## Fusion Configuration (`fusion_config.py`)

Edit this file to tune statistical behaviour without touching algorithm code.

```python
ABSOLUTE_THRESHOLD = {
    "molecular_weight": 50.0, "exact_mol_weight": 0.005,
    "alogp": 0.5, "molecular_polar_surface_area": 10.0,
}
PLAUSIBILITY_FLOOR = {
    "number_of_atoms": 1, "num_rotatable_bonds": 0,
    "molecular_weight": 10.0, "exact_mol_weight": 10.0,
    "molecular_polar_surface_area": 0.0,
}
PLAUSIBILITY_CEILING = {
    "molecular_weight": 2000.0, "exact_mol_weight": 2000.0,
    "num_h_acceptors_lipinski": 20, "num_h_donors_lipinski": 10,
    "num_h_acceptors": 20, "num_h_donors": 10,
    "num_rotatable_bonds": 50, "molecular_polar_surface_area": 500.0,
    "alogp": 10.0, "number_of_atoms": 500,
}
CONFIDENCE_CAP = {
    "alogp": 0.70, "molecular_weight": 0.80, "exact_mol_weight": 0.85,
    "molecular_polar_surface_area": 0.75, "number_of_atoms": 0.80,
    "num_h_acceptors_lipinski": 0.80, "num_h_donors_lipinski": 0.80,
    "num_h_acceptors": 0.80, "num_h_donors": 0.80,
    "num_rotatable_bonds": 0.80, "net_formal_charge": 0.85,
}
```

---

## Guards & Safety Mechanisms

| Guard | Location | Prevents |
|---|---|---|
| Non-resolution contract | LLM1 system prompt | LLM averaging or resolving conflicts before fusion |
| Confidence gating (0.4) | fusion.py | Low-quality candidates polluting fusion |
| Source-priority boosting | fusion.py | Literature sources dominating PubChem/DrugBank values |
| Plausibility floor/ceiling | fusion.py + fusion_config.py | Physically impossible values entering the pool |
| Consensus hallucination detection | fusion.py | Zero-variance pools from model echoing one number |
| CAS canonical selection | fusion.py | Salt/polymorph CAS replacing the primary |
| Numeric re-injection `[GUARD]` | main.py | LLM2 silently altering fused numeric values |
| String-field guard | main.py | LLM2 overwriting non-fillable resolved fields |
| Composition dict guard | main.py | LLM2 losing or clearing the fused composition dict |
| SMILES validation `[SMILES]` | main.py | Invalid SMILES reaching the output |
| Deterministic recomputation `[RECOMPUTE]` | main.py | LLM noise in MW, exact mass, HAC, composition when formula is known |
| Formula↔MW check `[CHEM-CHECK]` | main.py | Formula/MW mismatch >1 Da; overwrites with formula-derived value |
| SMILES↔HAC check `[CHEM-CHECK]` | main.py | SMILES heavy-atom count inconsistent with formula |
| Composition fraction check `[COMP]` | main.py | Fractions not summing to 1.0; renormalises in place |
| Integer enforcement `[INT-GUARD]` | main.py | LLM2 producing floats for integer fields |
| Null sentinel overwrite `[INFO]` | main.py | LLM2 replacing an honest `null` with a schema default |
| Schema key audit `[WARN]` | main.py | LLM2 silently dropping required fields |
| Duplicate accession guard | fusion.py + main.py | `drugbank_id` appearing in `secondary_accession_numbers` |

---

## Quickstart

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running locally, `deepseek-r1:70b` pulled (needs ~48GB VRAM for full precision — a dual RTX 4090 setup works).

```bash
git clone https://github.com/your-username/biomedical-property-pipeline.git
cd biomedical-property-pipeline
pip install ollama requests json-repair
ollama pull deepseek-r1:70b
```

No `.env` or API key needed — `main.py` connects to the local Ollama server via `ollama.Client()`.

Run the pipeline interactively:

```bash
python main.py
# Enter molecule name OR CID: Ibuprofen
# Debug mode? (y/n): n
```

Accepted inputs: common name (`Ibuprofen`), IUPAC name, or PubChem CID (`5754`). Debug mode prints the raw LLM1 extraction and fused intermediate before LLM2 runs.

---

## Batch Processing (`Generate_drugbank.py`)

Reads `Drugbank.csv`, runs the pipeline for up to 50 rows with a non-zero CID, and writes results to `Drugbank_Generated.csv`. Resumes automatically from where it left off if interrupted.

```bash
python Generate_drugbank.py
```

List fields are expanded into fixed columns (`Synonym_1/2/3`, `Secondary_Accession_1/2`). `molecular_composition` is serialised as a single string cell (e.g. `C: 0.6008, H: 0.0446, O: 0.3546`).

### Preparing `Drugbank.csv` (`db2csv.py`)

If `Drugbank.csv` is missing its `CID` column, run `db2csv.py` first. It resolves each `DRUGBANK_ID` to a PubChem CID via the PubChem REST API using 7 parallel workers, flushes progress every 100 rows, and resumes automatically on re-run.

```bash
python db2csv.py
```

### Comparing Against Ground Truth (`compare.py`)

Compares `Drugbank_Generated.csv` against `Drugbank.csv` field-by-field, aligning rows on shared `CID`. Numeric fields report bias, median error, MAE, std, and outlier count (`|err| > 2σ`); string fields report exact-match percentage (with normalisation for formulas, names, and element compositions).

```bash
python compare.py
```

---

## Output Format

`output.json` is written after every successful run. All floats are rounded to 4 decimal places. Fields set to `null` indicate the fusion engine could not resolve a plausible value — this is intentional, not a bug.

Sample output for Ibuprofen (CID 5754):

```json
{
  "cid": 5754,
  "drugbank_id": "DB01050",
  "secondary_accession_numbers": ["DBSALT001050"],
  "common_name": "Ibuprofen",
  "cas_number": "15687-27-1",
  "unii": "WK2XYI10QM",
  "synonyms": ["Advil", "Motrin", "2-(4-isobutylphenyl)propanoic acid"],
  "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
  "number_of_atoms": 15,
  "net_formal_charge": 0,
  "molecular_formula": "C13H18O2",
  "molecular_composition": {"C": 0.7771, "H": 0.0902, "O": 0.1591},
  "molecular_weight": 206.2808,
  "exact_mol_weight": 206.1307,
  "num_h_acceptors_lipinski": 2,
  "num_h_donors_lipinski": 1,
  "num_rotatable_bonds": 4,
  "num_h_acceptors": 2,
  "num_h_donors": 1,
  "alogp": 3.97,
  "molecular_polar_surface_area": 37.3
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local Ollama Python client for LLM1 and LLM2 (`client.chat(model="deepseek-r1:70b", ...)`) |
| `requests` | PubChem REST API calls (CID resolution + name enrichment); also used in `db2csv.py` |
| `json-repair` | Recovers truncated/malformed JSON from LLM output as a last-resort parse fallback |
| `pandas` | Row alignment and column comparison in `compare.py` |
| `numpy` | Numeric error stats (bias, MAE, std, outliers) in `compare.py` |
| `statistics` | `median()` for outlier filtering in fusion |
| `collections` | `defaultdict` for fusion bucketing and composition accumulation |
| `json`, `re`, `math`, `time`, `os`, `csv` | Standard library |

```bash
pip install ollama requests json-repair pandas numpy
```

External requirement: an Ollama server running locally with `deepseek-r1:70b` pulled (`ollama pull deepseek-r1:70b`).

---

## Extending the Schema

To add a new field (e.g. `tpsa`):

1. **`drug_descriptors.py`** — add to `REQUIRED_SCHEMA` with a typed placeholder, `FIELD_TYPES`, and optionally `DETERMINISTIC_FIELDS`.
2. **`fusion_config.py`** — add entries to `ABSOLUTE_THRESHOLD`, `PLAUSIBILITY_FLOOR`/`CEILING`, and `CONFIDENCE_CAP` as needed.

`main.py`, `LLM1.py`, `LLM2.py`, and `fusion.py` all derive behaviour from the schema at runtime — no other changes required.

---

## Known Limitations

- **Model hallucination** — registry fields like `drugbank_id` and `unii` require exact lookups; the model can fabricate plausible-sounding but incorrect values for obscure molecules. Confidence gating and plausibility windows mitigate but cannot eliminate this.
- **LLM1 integer compliance** — despite explicit prompt instructions, the model occasionally emits floats for integer fields. `[INT-GUARD]` is the last-resort catch.
- **SMILES validation without RDKit** — the syntactic check in `main.py` cannot detect chemically invalid SMILES that pass character-level validation. Full validation requires RDKit.
- **Single-candidate pools** — if only one candidate survives the confidence filter, outlier removal has nothing to compare against and accepts it unconditionally.
- **Local inference latency** — `deepseek-r1:70b` thinks by default (emits `<think>` blocks before JSON), which adds output tokens and latency per call; `num_ctx`/`num_predict` are tuned per-file to budget for this.
- **PubChem CID fallback** — if name lookup fails, `cid` is set to `0` (a sentinel, not a valid PubChem record). Downstream consumers should treat `cid == 0` as unresolved.
- **`secondary_accession_numbers`** — not guard-protected beyond deduplication; LLM2 may add or reorder entries freely.