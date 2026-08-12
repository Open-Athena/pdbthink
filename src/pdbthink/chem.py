"""Versioned chemistry constants.

Everything here is static reference data that affects gold labels, so each table
carries a version string that is recorded in instance provenance:

* ``CCD_LITE_VERSION``  - covalent bond dictionary (Appendix A.8)
* ``RADII_VERSION``     - van der Waals radii (A.13 SASA, A.18 clashes)
* ``MAX_ASA_VERSION``   - residue-specific maximum ASA (A.13)
"""

from __future__ import annotations

from .util import stable_hash

CCD_LITE_VERSION = "ccd_lite_v1"
RADII_VERSION = "bondi_pdbthink_v1"
MAX_ASA_VERSION = "tien_2013_theoretical"

# --------------------------------------------------------------------------- #
# Residue identity
# --------------------------------------------------------------------------- #

STANDARD_AA = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

#: Modified residues retained inside the polymer. The identifier uses the parent
#: amino-acid letter (A.5) but the component code stays visible in coordinates.
PARENT_RESIDUE = {
    "MSE": "MET",   # selenomethionine
    "SEP": "SER",   # phosphoserine
    "TPO": "THR",   # phosphothreonine
    "PTR": "TYR",   # phosphotyrosine
    "CSO": "CYS",   # S-hydroxycysteine
    "CME": "CYS",   # S,S-(2-hydroxyethyl)thiocysteine
    "CSD": "CYS",
    "OCS": "CYS",
    "KCX": "LYS",   # lysine NZ-carboxylic acid
    "LLP": "LYS",   # lysine-pyridoxal phosphate
    "M3L": "LYS",
    "MLY": "LYS",
    "ALY": "LYS",
    "HYP": "PRO",
    "PCA": "GLN",   # pyroglutamate
    "TYS": "TYR",
    "HIC": "HIS",
    "MLZ": "LYS",
    "SAC": "SER",
    "FME": "MET",
    "DDE": "HIS",
    "NEP": "HIS",
}

DNA_RESIDUES = frozenset({"DA", "DC", "DG", "DT", "DI", "DU"})
RNA_RESIDUES = frozenset({"A", "C", "G", "U", "I", "N"})

PHOSPHO_COMPONENTS = frozenset({"SEP", "TPO", "PTR"})

BACKBONE_ATOMS = ("N", "CA", "C", "O")


def three_to_one(resname: str) -> str:
    """One-letter code for a (possibly modified) amino acid, ``X`` if unknown."""
    name = resname.strip().upper()
    if name in AA3_TO_1:
        return AA3_TO_1[name]
    parent = PARENT_RESIDUE.get(name)
    if parent:
        return AA3_TO_1.get(parent, "X")
    return "X"


def is_amino_acid(resname: str) -> bool:
    name = resname.strip().upper()
    return name in AA3_TO_1 or name in PARENT_RESIDUE


def parent_of(resname: str) -> str:
    """Standard parent component code, used to look up bonds and chi1 atoms."""
    name = resname.strip().upper()
    return PARENT_RESIDUE.get(name, name)


# --------------------------------------------------------------------------- #
# van der Waals radii (Bondi 1964, with common metal values); Angstrom
# --------------------------------------------------------------------------- #

VDW_RADII = {
    "H": 1.20, "D": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
    "ZN": 1.39, "MG": 1.73, "MN": 1.61, "FE": 1.56, "CU": 1.40, "CO": 1.53,
    "NI": 1.63, "CA": 1.95, "NA": 2.27, "K": 2.75, "CD": 1.58, "HG": 1.55,
    "B": 1.92, "SI": 2.10, "AS": 1.85, "LI": 1.82, "PT": 1.75, "AU": 1.66,
}
DEFAULT_VDW_RADIUS = 1.70


def vdw_radius(element: str) -> float:
    return VDW_RADII.get(element.strip().upper(), DEFAULT_VDW_RADIUS)


# --------------------------------------------------------------------------- #
# Tien et al. 2013 theoretical maximum ASA, square Angstrom
# --------------------------------------------------------------------------- #

MAX_ASA = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}


def max_asa(resname: str) -> float | None:
    return MAX_ASA.get(parent_of(resname))


# --------------------------------------------------------------------------- #
# Metals (A.19)
# --------------------------------------------------------------------------- #

MONATOMIC_METALS = frozenset(
    {"ZN", "MG", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CA", "NA", "K", "CD", "HG"}
)

#: Element symbol for monatomic metal components whose code is not the element.
METAL_ELEMENT = {"FE2": "FE", "CU1": "CU", "3CO": "CO"}


def metal_element(resname: str) -> str:
    name = resname.strip().upper()
    return METAL_ELEMENT.get(name, name)


# --------------------------------------------------------------------------- #
# Covalent topology: ccd_lite_v1 (A.8)
# --------------------------------------------------------------------------- #
# Heavy-atom bonds only. Alternate atom-name spellings are listed as extra
# bonds; pairs referencing absent atoms are ignored when the graph is built.

_BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"))

_SIDECHAIN_BONDS: dict[str, tuple[tuple[str, str], ...]] = {
    "GLY": (),
    "ALA": (("CA", "CB"),),
    "SER": (("CA", "CB"), ("CB", "OG")),
    "CYS": (("CA", "CB"), ("CB", "SG")),
    "THR": (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
    "VAL": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
    "LEU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    "ILE": (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"), ("CG1", "CD")),
    "MET": (("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE")),
    "MSE": (("CA", "CB"), ("CB", "CG"), ("CG", "SE"), ("SE", "CE")),
    "PRO": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")),
    "PHE": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
    ),
    "TYR": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"), ("CZ", "OH"),
    ),
    "TRP": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
        ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"),
        ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2"),
    ),
    "HIS": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"),
        ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2"),
    ),
    "ASP": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
    "ASN": (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
    "GLU": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
    "GLN": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
    "LYS": (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
    "ARG": (
        ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
        ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
    ),
}

_PHOSPHATE_BONDS = (
    ("P", "O1P"), ("P", "O2P"), ("P", "O3P"),
    ("P", "OP1"), ("P", "OP2"), ("P", "OP3"),
)

_MODIFIED_EXTRA: dict[str, tuple[tuple[str, str], ...]] = {
    "SEP": (("OG", "P"),) + _PHOSPHATE_BONDS,
    "TPO": (("OG1", "P"),) + _PHOSPHATE_BONDS,
    "PTR": (("OH", "P"),) + _PHOSPHATE_BONDS,
    "CSO": (("SG", "OD"),),
    "HYP": (("CG", "OD1"),),
    "KCX": (("NZ", "CX"), ("CX", "OQ1"), ("CX", "OQ2")),
    "MLY": (("NZ", "CH1"), ("NZ", "CH2")),
    "M3L": (("NZ", "CM1"), ("NZ", "CM2"), ("NZ", "CM3")),
    "ALY": (("NZ", "CH"), ("CH", "OH"), ("CH", "CH3")),
    "TYS": (("OH", "S"), ("S", "O1"), ("S", "O2"), ("S", "O3")),
}


def component_bonds(resname: str) -> tuple[tuple[str, str], ...]:
    """Intra-residue heavy-atom bonds for a standard or supported residue.

    Returns an empty tuple for components that are not in the dictionary; the
    caller must then treat the component as having unknown topology (A.8
    restricts nearest-nonbonded questions to standard residues).
    """
    name = resname.strip().upper()
    parent = parent_of(name)
    if parent not in _SIDECHAIN_BONDS and name not in _SIDECHAIN_BONDS:
        return ()
    side = _SIDECHAIN_BONDS.get(name) or _SIDECHAIN_BONDS.get(parent, ())
    extra = _MODIFIED_EXTRA.get(name, ())
    return _BACKBONE_BONDS + side + extra


def has_known_topology(resname: str) -> bool:
    name = resname.strip().upper()
    return name in _SIDECHAIN_BONDS or parent_of(name) in _SIDECHAIN_BONDS


#: Atoms whose presence is required to compute chi1 (A.21).
CHI1_ATOM4 = {
    "SER": "OG", "THR": "OG1", "CYS": "SG", "VAL": "CG1", "ILE": "CG1",
}
CHI1_EXCLUDED = frozenset({"GLY", "ALA", "PRO"})


def chi1_atom4(resname: str) -> str | None:
    """Fourth atom of the ``N-CA-CB-XG`` chi1 dihedral, or ``None`` if excluded."""
    parent = parent_of(resname)
    if parent in CHI1_EXCLUDED or parent not in AA3_TO_1:
        return None
    return CHI1_ATOM4.get(parent, "CG")


#: Aromatic ring atom sets used by A.32 (curated mechanistic episodes only).
AROMATIC_RINGS = {
    "PHE": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TYR": (("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),),
    "TRP": (
        ("CG", "CD1", "NE1", "CE2", "CD2"),
        ("CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"),
    ),
    "HIS": (("CG", "ND1", "CE1", "NE2", "CD2"),),
}

#: Charged atoms for salt bridges (A.11).
SALT_BRIDGE_POSITIVE = {"LYS": ("NZ",), "ARG": ("NE", "NH1", "NH2")}
SALT_BRIDGE_NEGATIVE = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}


def chemistry_fingerprint() -> str:
    """Content hash over every table that can change a gold label."""
    return stable_hash(
        CCD_LITE_VERSION,
        RADII_VERSION,
        MAX_ASA_VERSION,
        sorted(VDW_RADII.items()),
        sorted(MAX_ASA.items()),
        sorted((k, sorted(v)) for k, v in _SIDECHAIN_BONDS.items()),
        sorted((k, sorted(v)) for k, v in _MODIFIED_EXTRA.items()),
        sorted(PARENT_RESIDUE.items()),
    )[:16]
