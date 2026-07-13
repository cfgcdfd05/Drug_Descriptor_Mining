"""
LLM1.py — Multi-candidate raw extraction.

Model: deepseek-r1:70b (thinks by default — think=True not needed/supported)
  - format="json" REMOVED — can conflict with R1 thinking tokens
  - num_predict base 5120, scaling +1024 per retry
  - num_ctx 6144 — budgets ~355 input + ~2000 thinking + ~900 output
  - clean_json_output strips <think> blocks correctly
"""

import json
import re
import time


def _scrub(text: str) -> str:
    """Strip BOM and ASCII control characters (except tab/newline)."""
    text = text.lstrip('\ufeff')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def clean_json_output(text: str) -> dict:
    # Strip Qwen3 thinking tokens (open-ended pattern catches unclosed tags too)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*',          '', text, flags=re.DOTALL)
    text = _scrub(text)

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block fallback
    matches = re.findall(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if matches:
        candidate = matches[-1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    else:
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else text

    # 3. json-repair — recovers truncated output
    try:
        from json_repair import repair_json
        repaired = repair_json(candidate)
        if repaired:
            result = json.loads(repaired)
            print("[LLM1] Used json-repair to recover malformed output.")
            return result
    except Exception:
        pass

    raise ValueError("No JSON found in LLM1 output.")


_SCHEMA_COMPACT = """\
drugbank_id, secondary_accession_numbers, common_name, cas_number, unii,
synonyms, smiles, net_formal_charge, num_h_acceptors_lipinski,
num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors,
alogp, molecular_polar_surface_area, molecular_formula, molecular_weight,
exact_mol_weight, number_of_atoms, molecular_composition"""


_SYSTEM = """\
Extract drug properties. Your entire response must be a single valid JSON object.
No preamble, no explanation, no markdown fences — only the JSON.

Every field must be a LIST of candidate dicts.

FORMAT — scalar fields:
  "field": [{"value": <v>, "confidence": <0.0-1.0>}]

FORMAT — list-valued fields (secondary_accession_numbers, synonyms):
  "synonyms": [{"value": ["aspirin", "acetylsalicylic acid"], "confidence": 0.9}]

FORMAT — composition field:
  "molecular_composition": [{"value": {"C": 0.600, "H": 0.045, "O": 0.355}, "confidence": 0.9}]

RULES:
- drugbank_id / cas_number / unii / common_name: registry values only; omit if unsure.
- Never put drugbank_id inside secondary_accession_numbers.
- smiles: canonical SMILES only if certain; else "".
- net_formal_charge: integer (0 for neutral).
- num_h_acceptors_lipinski, num_h_donors_lipinski, num_rotatable_bonds, num_h_acceptors, num_h_donors: integers.
- alogp, molecular_polar_surface_area, molecular_weight, exact_mol_weight: floats.
- molecular_composition fractions must sum to ~1.0.

FIELDS: """ + _SCHEMA_COMPACT


def run_extraction(
    molecule_input: str,
    cid: int,
    schema: dict,
    client,
    retries: int = 3,
    wait: int = 1,
) -> dict:
    cid_hint = f" (PubChem CID: {cid})" if cid else ""
    user_content = f"Extract properties for: {molecule_input}{cid_hint}"

    for attempt in range(1, retries + 1):
        num_predict = 5120 + (attempt - 1) * 1024
        try:
            result = client.chat(
                model="deepseek-r1:70b",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                # think=True not needed — deepseek-r1 thinks by default
                # format="json" intentionally absent — can conflict with R1 thinking tokens
                options={
                    "temperature":    0.0,
                    "top_k":          1,
                    "top_p":          1.0,
                    "repeat_penalty": 1.0,
                    "num_predict":    num_predict,
                    "num_ctx":        4096,
                    "seed":           42,
                },
            )
            raw_text = result["message"]["content"]
            return clean_json_output(raw_text)

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} malformed JSON ({e}). "
                      f"Retrying with num_predict={num_predict + 1024}...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM1 malformed JSON after {retries} attempts: {e}")

        except Exception as e:
            if attempt < retries:
                print(f"[LLM1] Attempt {attempt} failed ({type(e).__name__}: {e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM1 failed after {retries} attempts: {e}")