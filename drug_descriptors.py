# ---------------------------------------------------------------------------
# REQUIRED_SCHEMA
# Flat structure matching CSV columns exactly.
# Default values encode the *type* each leaf must hold after fusion/LLM2:
#   int   → 0          float → 0.0         str → ""     list → []
#   dict  → {}         None-able fields use None as sentinel elsewhere
# ---------------------------------------------------------------------------

REQUIRED_SCHEMA = {
    "cid":                          0,

    # Identity
    "drugbank_id":                  "",
    "secondary_accession_numbers":  [],
    "common_name":                  "",
    "cas_number":                   "",
    "unii":                         "",
    "synonyms":                     [],
    "smiles":                       "",

    # Molecule — integers
    "number_of_atoms":              0,      # integer — count of all non-H atoms (was number_of_heavy_atoms)
    "net_formal_charge":            0,      # integer

    # Molecule — strings/specials
    "molecular_formula":            "",
    "molecular_composition":        {},     # {element: mass_fraction} e.g. {"C": 0.600, "H": 0.045, "O": 0.355}

    # Molecule — floats
    "molecular_weight":             0.0,
    "exact_mol_weight":             0.0,

    # Molecule — integers (Lipinski / descriptor counts)
    "num_h_acceptors_lipinski":     0,      # integer
    "num_h_donors_lipinski":        0,      # integer
    "num_rotatable_bonds":          0,      # integer
    "num_h_acceptors":              0,      # integer
    "num_h_donors":                 0,      # integer

    # Molecule — float descriptors
    "alogp":                        0.0,
    "molecular_polar_surface_area": 0.0,
}

# ---------------------------------------------------------------------------
# FIELD_TYPES
# Controls which fusion strategy is applied to each leaf field.
# ---------------------------------------------------------------------------

FIELD_TYPES = {
    # Identity
    "drugbank_id":                  "categorical",
    "secondary_accession_numbers":  "list",
    "common_name":                  "categorical",
    "cas_number":                   "categorical",
    "unii":                         "categorical",
    "synonyms":                     "list",
    "smiles":                       "smiles",

    # Molecule — integers
    "number_of_atoms":              "integer",
    "net_formal_charge":            "integer",
    "num_h_acceptors_lipinski":     "integer",
    "num_h_donors_lipinski":        "integer",
    "num_rotatable_bonds":          "integer",
    "num_h_acceptors":              "integer",
    "num_h_donors":                 "integer",

    # Molecule — floats
    "molecular_weight":             "float",
    "exact_mol_weight":             "float",
    "alogp":                        "float",
    "molecular_polar_surface_area": "float",

    # Molecule — specials
    "molecular_formula":            "categorical",
    "molecular_composition":        "composition",
}

# ---------------------------------------------------------------------------
# SOURCE_PRIORITY
# Higher number = higher trust.
# ---------------------------------------------------------------------------

SOURCE_PRIORITY = {
    "pubchem":    5,
    "drugbank":   5,
    "chembl":     4,
    "bindingdb":  3,
    "literature": 2,
    "other":      1,
}

def source_priority(source_type: str) -> int:
    if not isinstance(source_type, str):
        return 1
    key = source_type.strip().lower()
    for name, score in SOURCE_PRIORITY.items():
        if name in key:
            return score
    return 1


# ---------------------------------------------------------------------------
# DETERMINISTIC_FIELDS
# ---------------------------------------------------------------------------

DETERMINISTIC_FIELDS = {
    "number_of_atoms",
    "molecular_weight",
    "exact_mol_weight",
    "molecular_formula",
    "molecular_composition",
    "net_formal_charge",
    "num_h_acceptors_lipinski",
    "num_h_donors_lipinski",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "molecular_polar_surface_area",
}