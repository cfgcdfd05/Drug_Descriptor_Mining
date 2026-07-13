# ---------------------------------------------------------------------------
# fusion_config.py
# ---------------------------------------------------------------------------

# Absolute deviation from median allowed before a candidate is treated as an
# outlier.  Only applied to float fields where ratio-based filtering is too
# coarse (e.g. MW, TPSA, exact mass).
ABSOLUTE_THRESHOLD = {
    "molecular_weight":             50.0,
    "exact_mol_weight":             0.005,
    "alogp":                        0.5,
    "molecular_polar_surface_area": 10.0,
    # clean_energy removed — no agreed units / method, extraction unreliable
}

# Values strictly below the floor are implausible and discarded.
PLAUSIBILITY_FLOOR = {
    "number_of_atoms":          1,
    "num_rotatable_bonds":      0,
    "molecular_weight":         10.0,
    "exact_mol_weight":         10.0,
    "molecular_polar_surface_area": 0.0,
}

# Values strictly above the ceiling are implausible and discarded.
PLAUSIBILITY_CEILING = {
    "molecular_weight":             2000.0,
    "exact_mol_weight":             2000.0,
    "num_h_acceptors_lipinski":     20,
    "num_h_donors_lipinski":        10,
    "num_h_acceptors":              20,
    "num_h_donors":                 10,
    "num_rotatable_bonds":          50,
    "molecular_polar_surface_area": 500.0,
    "alogp":                        10.0,
    "number_of_atoms":              500,
}

# Per-field confidence caps applied AFTER source-priority boosting.
# Deterministic descriptors extracted via LLM are capped lower because the
# ground truth is structural, not literature-measured.
CONFIDENCE_CAP = {
    # Deterministic — LLM is a fallback, not an authority
    "alogp":                        0.70,
    "molecular_weight":             0.80,
    "exact_mol_weight":             0.85,
    "molecular_polar_surface_area": 0.75,
    "number_of_atoms":          0.80,
    "num_h_acceptors_lipinski":     0.80,
    "num_h_donors_lipinski":        0.80,
    "num_h_acceptors":              0.80,
    "num_h_donors":                 0.80,
    "num_rotatable_bonds":          0.80,
    "net_formal_charge":            0.85,
}

# Fields that are purely experimental / literature-measured.
# For these, identical values across independent sources ARE expected
# (same assay result), so consensus-hallucination detection is disabled.
EXPERIMENTAL_FIELDS: set = set()   # no experimental fields in current schema

# Fields whose values are continuous and amenable to confidence-weighted mean.
# Integer fields are NOT listed here — they use weighted mode → round.
ACTIVITY_FIELDS: set = set()       # no activity fields in current schema