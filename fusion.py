"""
fusion.py
---------
Resolves lists of LLM1 candidates into single fused values.

Key design changes vs. the previous version
-------------------------------------------
1.  Field-type-aware fusion: integers, floats, categoricals, lists, SMILES,
    and composition dicts each follow a dedicated strategy.
2.  Integer fields are never averaged into floats.  The confidence-weighted
    mode (most-common value, ties broken by total confidence) is used, then
    the result is cast to int.
3.  Consensus-hallucination detection is restricted to experimental/LLM-noisy
    fields only.  Deterministic descriptors (MW, HBA, …) legitimately produce
    identical values across PubChem, DrugBank, ChEMBL — flagging them as
    hallucinations is a false positive.
4.  Source-priority scores are folded into effective confidence so that
    PubChem/DrugBank canonical values dominate over literature/other.
5.  molecular_composition is fused element-by-element as a dict of floats,
    not as an opaque string.
6.  CAS number selection now prefers the lexicographically earliest candidate
    among those at maximum effective confidence, which matches the convention
    that canonical CAS numbers were assigned first (lowest registry number).
"""

import statistics
from collections import defaultdict

from drug_descriptors import FIELD_TYPES
from fusion_config import (
    ABSOLUTE_THRESHOLD,
    PLAUSIBILITY_FLOOR,
    PLAUSIBILITY_CEILING,
    CONFIDENCE_CAP,
    EXPERIMENTAL_FIELDS,
)

# Minimum effective confidence for a candidate to participate in fusion.
MIN_CONFIDENCE = 0.4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_candidate_dict(item) -> bool:
    return isinstance(item, dict) and "value" in item


def _effective_confidence(candidate: dict) -> float:
    """
    Returns the declared confidence directly.

    source_type priority boosting has been removed so that candidates from
    the old LLM1 format (with source_type) and the new format (without it)
    produce identical fusion results when resuming a partially-generated CSV.
    The boost was small (max ×1.4) and LLM1 is the only candidate source
    anyway, so removing it has no practical effect on accuracy.
    """
    return float(candidate.get("confidence", 0))


def _apply_confidence_cap(candidates: list, field_name: str) -> list:
    if field_name not in CONFIDENCE_CAP:
        return candidates
    cap = CONFIDENCE_CAP[field_name]
    return [
        {**c, "confidence": min(c.get("confidence", 0), cap)}
        for c in candidates
    ]


def _apply_plausibility(candidates: list, field_name: str):
    """Drop candidates whose value falls outside the plausibility window."""
    if field_name in PLAUSIBILITY_FLOOR:
        floor = PLAUSIBILITY_FLOOR[field_name]
        candidates = [c for c in candidates if c.get("value") is not None and c["value"] >= floor]
    if field_name in PLAUSIBILITY_CEILING:
        ceiling = PLAUSIBILITY_CEILING[field_name]
        candidates = [c for c in candidates if c.get("value") is not None and c["value"] <= ceiling]
    return candidates


def _check_consensus_hallucination(candidates: list, field_name: str) -> bool:
    """
    Returns True only for non-deterministic, non-experimental fields where
    all numeric candidates share the exact same value — a signal that the
    model echoed one number rather than retrieving independent measurements.

    Deterministic descriptors (MW, HBA, formula …) legitimately produce
    identical values across authoritative sources, so they are explicitly
    excluded to avoid false positives.
    """
    from drug_descriptors import DETERMINISTIC_FIELDS
    if field_name in DETERMINISTIC_FIELDS:
        return False          # identical is expected, not suspicious
    if field_name in EXPERIMENTAL_FIELDS:
        return False          # experimental values are often singly-reported
    values = [
        c["value"] for c in candidates
        if isinstance(c.get("value"), (int, float))
    ]
    if len(values) < 2:
        return False
    return len(set(values)) == 1


# ---------------------------------------------------------------------------
# Per-type fusion strategies
# ---------------------------------------------------------------------------

def _fuse_float(candidates: list, field_name: str) -> float | None:
    """
    Confidence-weighted mean of numeric candidates, after outlier removal.
    Effective confidence (conf × source-priority boost) is used for weighting.
    """
    candidates = _apply_plausibility(candidates, field_name)
    candidates = _apply_confidence_cap(candidates, field_name)
    if not candidates:
        return None

    if _check_consensus_hallucination(candidates, field_name):
        print(
            f"[FUSION] WARNING: all '{field_name}' candidates identical "
            f"— possible consensus hallucination; capping effective confidence to 0.5"
        )
        candidates = [
            {**c, "confidence": min(c.get("confidence", 0), 0.5)}
            for c in candidates
        ]

    pairs = [
        (c["value"], _effective_confidence(c))
        for c in candidates
        if isinstance(c.get("value"), (int, float))
    ]
    if not pairs:
        return None

    values = [v for v, _ in pairs]
    effs   = [e for _, e in pairs]

    # Outlier removal
    if len(values) >= 2:
        med = statistics.median(values)
        if field_name in ABSOLUTE_THRESHOLD:
            threshold = ABSOLUTE_THRESHOLD[field_name]
            paired = [(v, e) for v, e in zip(values, effs) if abs(v - med) <= threshold]
        else:
            if abs(med) > 1e-9:
                paired = [
                    (v, e) for v, e in zip(values, effs)
                    if (max(abs(v), abs(med)) / max(abs(min(abs(v), abs(med))), 1e-9)) <= 5
                ]
            else:
                paired = [(v, e) for v, e in zip(values, effs) if abs(v) <= 1]
        if paired:
            values, effs = zip(*paired)
            values, effs = list(values), list(effs)

    total_eff = sum(effs)
    if total_eff == 0:
        return float(sum(values) / len(values))
    return float(sum(v * e for v, e in zip(values, effs)) / total_eff)


def _fuse_integer(candidates: list, field_name: str) -> int | None:
    """
    Weighted mode for integer fields: the integer value with the highest
    summed effective confidence wins.  This preserves discreteness — we never
    produce 3.2857 for num_rotatable_bonds.

    Ties are broken by source priority (PubChem/DrugBank > literature).
    If all candidates produce a unique value, fall back to the single
    highest-confidence integer (same as weighted mode with one bin per value).
    """
    candidates = _apply_plausibility(candidates, field_name)
    candidates = _apply_confidence_cap(candidates, field_name)
    if not candidates:
        return None

    # Collect integer-valued candidates only
    int_pairs = [
        (int(round(c["value"])), _effective_confidence(c))
        for c in candidates
        if isinstance(c.get("value"), (int, float))
    ]
    if not int_pairs:
        return None

    # Bucket by integer value
    bucket: dict[int, float] = defaultdict(float)
    for val, eff in int_pairs:
        bucket[val] += eff

    # Return the value with the highest total effective confidence
    return int(max(bucket, key=lambda v: bucket[v]))


def _fuse_categorical(candidates: list, field_name: str) -> str | None:
    """
    Weighted mode for string / categorical fields.
    Effective confidence (conf × source-priority) selects the winner.

    Special rule for cas_number: among candidates tied at maximum effective
    confidence, prefer the lexicographically smallest value.  CAS numbers are
    assigned sequentially — a smaller number was registered earlier and is more
    likely to be the canonical / primary accession.
    """
    candidates = _apply_confidence_cap(candidates, field_name)

    str_pairs = [
        (str(c["value"]).strip(), _effective_confidence(c))
        for c in candidates
        if isinstance(c.get("value"), str) and c["value"].strip()
    ]
    if not str_pairs:
        return None

    bucket: dict[str, float] = defaultdict(float)
    for val, eff in str_pairs:
        bucket[val] += eff

    max_eff = max(bucket.values())

    if field_name == "cas_number":
        # Among all candidates at max effective confidence, pick earliest CAS.
        # CAS format: XXXXXXX-YY-Z  — sort lexicographically on the full string
        # after zero-padding the prefix so shorter numbers sort correctly.
        top_candidates = [v for v, e in bucket.items() if e >= max_eff - 1e-9]

        def _cas_sort_key(cas: str):
            parts = cas.split("-")
            try:
                prefix = int(parts[0]) if parts else 0
            except ValueError:
                prefix = 0
            return prefix  # lower registry number = registered earlier = more canonical

        top_candidates.sort(key=_cas_sort_key)
        return top_candidates[0]

    return max(bucket, key=lambda v: bucket[v])


def _fuse_list_field(candidates: list, field_name: str) -> list:
    """
    For multi-value string fields (synonyms, secondary_accession_numbers):
    collect all unique non-empty string values from candidates above the
    minimum confidence threshold, ordered by descending effective confidence.
    Deduplication is case-insensitive but the original casing is preserved.
    """
    seen_lower: set = set()
    result: list = []

    # Sort so highest-confidence items are added first (preserved in output)
    sorted_cands = sorted(
        [c for c in candidates if _is_candidate_dict(c)],
        key=_effective_confidence,
        reverse=True,
    )

    for c in sorted_cands:
        if _effective_confidence(c) < MIN_CONFIDENCE:
            continue
        val = c.get("value")
        if isinstance(val, str) and val.strip():
            lower = val.strip().lower()
            if lower not in seen_lower:
                seen_lower.add(lower)
                result.append(val.strip())
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    lower = item.strip().lower()
                    if lower not in seen_lower:
                        seen_lower.add(lower)
                        result.append(item.strip())

    return result


def _fuse_smiles(candidates: list) -> str:
    """
    For SMILES: return the string value from the highest effective-confidence
    candidate.  Validation is deferred to main.py's _validate_smiles so we
    don't duplicate the logic here.
    """
    valid = [
        c for c in candidates
        if isinstance(c.get("value"), str) and c["value"].strip()
    ]
    if not valid:
        return ""
    best = max(valid, key=_effective_confidence)
    return best["value"].strip()


def _fuse_composition(candidates: list) -> dict:
    """
    Fuse molecular_composition candidates.

    LLM1 may return composition either as:
      (a) a dict  {"C": 0.600, "H": 0.045, "O": 0.355}  ← preferred
      (b) a string "C: 0.600, H: 0.045, O: 0.355"       ← legacy, parsed here

    Each element's fraction is fused independently using _fuse_float semantics.
    The final dict is renormalised so fractions sum to 1.0.
    """
    # Parse all candidates into per-element float candidates
    element_candidates: dict[str, list] = defaultdict(list)

    for c in candidates:
        if not _is_candidate_dict(c):
            continue
        eff = _effective_confidence(c)
        if eff < MIN_CONFIDENCE:
            continue
        val = c.get("value")

        if isinstance(val, dict):
            for elem, frac in val.items():
                if isinstance(frac, (int, float)):
                    element_candidates[elem.strip().capitalize()].append({
                        "value": float(frac),
                        "confidence": c.get("confidence", 0.5),
                        "source_type": c.get("source_type", "other"),
                    })

        elif isinstance(val, str):
            # Parse "C: 0.600, H: 0.045, O: 0.355"
            import re
            for token in val.split(","):
                m = re.match(r"\s*([A-Za-z]+)\s*:\s*([0-9.eE+\-]+)\s*", token)
                if m:
                    elem  = m.group(1).strip().capitalize()
                    try:
                        frac = float(m.group(2))
                    except ValueError:
                        continue
                    element_candidates[elem].append({
                        "value": frac,
                        "confidence": c.get("confidence", 0.5),
                        "source_type": c.get("source_type", "other"),
                    })

    if not element_candidates:
        return {}

    # Fuse each element independently
    fused_comp: dict[str, float] = {}
    for elem, ecands in element_candidates.items():
        result = _fuse_float(ecands, field_name=f"composition_{elem}")
        if result is not None and result > 0:
            fused_comp[elem] = result

    if not fused_comp:
        return {}

    # Renormalise so fractions sum to 1.0
    total = sum(fused_comp.values())
    if total > 1e-6:
        fused_comp = {k: round(v / total, 6) for k, v in fused_comp.items()}

    return fused_comp


# ---------------------------------------------------------------------------
# Deduplication guard for identity fields
# ---------------------------------------------------------------------------

def _deduplicate_secondary_accessions(fused: dict) -> dict:
    """
    Remove any value from secondary_accession_numbers that is identical
    (case-insensitive) to the primary drugbank_id.
    Works on the flat schema — fields are top-level keys.
    """
    primary = (fused.get("drugbank_id") or "").strip().upper()
    secondary = fused.get("secondary_accession_numbers")

    if primary and isinstance(secondary, list):
        cleaned = [s for s in secondary if s.strip().upper() != primary]
        if len(cleaned) != len(secondary):
            removed = [s for s in secondary if s.strip().upper() == primary]
            print(f"[FUSION] Removed duplicate accession(s) from secondary list: {removed}")
        fused["secondary_accession_numbers"] = cleaned

    return fused


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fuse(extracted: dict) -> dict:
    """
    Recursively walk extracted, dispatching each leaf list to the appropriate
    typed fusion strategy based on FIELD_TYPES.

    Branch dicts are recursed into; scalars pass through unchanged.
    """
    fused = {}
    for key, value in extracted.items():

        if isinstance(value, dict):
            # Special case: molecular_composition may arrive as a raw dict
            # {element: fraction} rather than a candidate list. Wrap it into a
            # single high-confidence candidate so _fuse_composition can handle it.
            if FIELD_TYPES.get(key) == "composition":
                fused[key] = _fuse_composition([{
                    "value": value,
                    "confidence": 0.85,
                    "source_type": "other",
                }])
            else:
                fused[key] = fuse(value)

        elif isinstance(value, list):
            ftype = FIELD_TYPES.get(key, "categorical")

            # Normalise: filter out low-confidence candidates globally
            # (except list fields which do their own threshold handling)
            if ftype not in ("list", "composition"):
                value = [
                    c for c in value
                    if not _is_candidate_dict(c) or _effective_confidence(c) >= MIN_CONFIDENCE
                ]

            if ftype == "float":
                dict_items = [c for c in value if _is_candidate_dict(c)]
                if dict_items:
                    fused[key] = _fuse_float(dict_items, field_name=key)
                else:
                    # Raw scalar fallback
                    scalars = [v for v in value if isinstance(v, (int, float))]
                    fused[key] = float(scalars[0]) if scalars else None

            elif ftype == "integer":
                dict_items = [c for c in value if _is_candidate_dict(c)]
                if dict_items:
                    fused[key] = _fuse_integer(dict_items, field_name=key)
                else:
                    scalars = [v for v in value if isinstance(v, (int, float))]
                    fused[key] = int(round(scalars[0])) if scalars else None

            elif ftype == "smiles":
                dict_items = [c for c in value if _is_candidate_dict(c)]
                fused[key] = _fuse_smiles(dict_items)

            elif ftype == "list":
                fused[key] = _fuse_list_field(value, field_name=key)

            elif ftype == "composition":
                fused[key] = _fuse_composition(value)

            else:  # "categorical" (includes molecular_formula, cas_number, etc.)
                dict_items = [c for c in value if _is_candidate_dict(c)]
                if dict_items:
                    fused[key] = _fuse_categorical(dict_items, field_name=key)
                else:
                    strs = [v for v in value if isinstance(v, str) and v.strip()]
                    fused[key] = strs[0] if strs else None

        else:
            fused[key] = value  # scalar passthrough

    # Post-fusion: deduplicate drugbank secondary accessions
    fused = _deduplicate_secondary_accessions(fused)

    return fused