"""
LLM2.py — Minimal gap-fill pass.
main.py guards protect all fused numeric values — LLM2 only fills empty fields.

Model: deepseek-r1:70b (thinks by default — think=True not needed/supported)
  - format="json" kept — Ollama JSON mode constrains output to bare JSON
  - Aggressive <think> strip kept as belt-and-suspenders for R1 bleedthrough
  - num_predict base 3072, scaling +1024 per retry up to 5120
  - num_ctx 4096 — budgets ~950 input + ~1500 thinking + ~600 output
  - Pre-parse scrub strips control characters and BOM
  - retries default 4
"""

import json
import re
import time


def _scrub(text: str) -> str:
    """Strip BOM, zero-width chars, and ASCII control chars (except tab/newline)."""
    text = text.lstrip('\ufeff')
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    return text.strip()


def _clean_json_output(text: str) -> dict:
    # Strip any Qwen3 thinking tokens that bled through
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*',          '', text, flags=re.DOTALL)  # unclosed tag
    text = _scrub(text)

    # 1. Direct parse — format="json" produces bare JSON
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
            print("[LLM2] Used json-repair to recover malformed output.")
            return result
    except Exception:
        pass

    raise ValueError("No JSON found in LLM2 output.")


_SYSTEM = """\
You receive fused drug data. Copy it exactly, then fill any field that is still at its \
default (0, 0.0, "", []).
Only fill a default field if you are certain of the correct value. Wrong is worse than empty.

RULES:
- Copy every non-default value EXACTLY. No rounding, no reformatting, no changes.
- CAS number: never substitute or invent. Copy as-is or leave empty.
- secondary_accession_numbers must NOT contain the drugbank_id value.
- smiles: fill only if 100% certain of the canonical structure; else keep "".
- INTEGER fields (must be whole numbers, no decimals ever): \
number_of_atoms, net_formal_charge, num_h_acceptors_lipinski, num_h_donors_lipinski, \
num_rotatable_bonds, num_h_acceptors, num_h_donors.
- FLOAT fields (must have a decimal point): \
molecular_weight, exact_mol_weight, alogp, molecular_polar_surface_area.
- molecular_composition: dict of {element: fraction} summing to ~1.0. Copy exactly if present.
- Output ONLY the schema fields plus an optional "warnings" list for any field you change.
- Output a valid JSON object with no extra keys."""


def run_verification(
    fused_data: dict,
    schema: dict,
    client,
    retries: int = 4,
    wait: int = 1,
) -> dict:
    user_content = (
        f"SCHEMA FIELDS: {list(schema.keys())}\n\n"
        f"FUSED DATA:\n{json.dumps(fused_data, indent=2)}"
    )

    for attempt in range(1, retries + 1):
        num_predict = min(3072 + (attempt - 1) * 1024, 5120)
        try:
            result = client.chat(
                model="deepseek-r1:70b",
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_content},
                ],
                # think=True not needed — deepseek-r1 thinks by default
                format="json",   # Ollama JSON mode — constrains output to bare JSON
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
            raw = result["message"]["content"]
            return _clean_json_output(raw)

        except (json.JSONDecodeError, ValueError) as e:
            if attempt < retries:
                next_budget = min(3072 + attempt * 1024, 5120)
                print(f"[LLM2] Attempt {attempt} malformed JSON ({e}). "
                      f"Retrying with num_predict={next_budget}...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 malformed JSON after {retries} attempts: {e}")

        except Exception as e:
            if attempt < retries:
                print(f"[LLM2] Attempt {attempt} failed ({type(e).__name__}: {e}). Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"LLM2 failed after {retries} attempts: {e}")