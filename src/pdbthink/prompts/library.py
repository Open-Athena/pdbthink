"""Versioned prompt text (specification section 7).

Changing anything in this module changes model-visible input and therefore
requires bumping :data:`PROMPT_VERSION`. :func:`prompt_fingerprint` hashes the
whole library so a dataset can prove which prompt text produced it.
"""

from __future__ import annotations

from ..util import stable_hash

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You will be given one or more molecular structures and a question about them.
Answer using only the information supplied in the prompt. Do not use tools or
external resources.

Protein residues are identified as <chain>:<one-letter amino-acid code><author
residue number>. For example, A:V22 is valine 22 in chain A. When asked for a
residue, use exactly this format. When asked for a list, return a comma-separated
list. Do not include additional residues.

Place the machine-readable answer after FINAL: using the format requested by
the question. You may reason before the final answer, but only the FINAL field
will be scored."""

#: Description of each model-visible representation.
REPRESENTATION_NOTICE = {
    "minimal_pdb": (
        "The structure is given below in PDB format. Only coordinate records are "
        "included; there are no header, annotation, connectivity or secondary-structure "
        "records, hydrogens have been removed, and B-factors are zero."
    ),
    "normalized_coordinates": (
        "The structure is given below as a comma-separated table with the columns "
        "record,chain,resnum,resname,atom,element,x,y,z. Coordinates are in Angstrom. "
        "Hydrogens have been removed."
    ),
    "context_only": (
        "No coordinates are provided for this question."
    ),
}

CROP_NOTICE = (
    "The coordinates below are a local structure excerpt, not the complete "
    "structure. They contain every atom needed to answer the question."
)

ROTATION_NOTICE = (
    "The coordinates have been rigidly rotated and translated; distances and "
    "angles are unchanged."
)

#: Answer-format instructions appended to each user prompt, keyed by answer schema.
FORMAT_INSTRUCTIONS = {
    "string_set": (
        "Answer with a comma-separated list of chain identifiers.\n"
        "Example: FINAL: A, B"
    ),
    "integer": "Answer with a single integer.\nExample: FINAL: 128",
    "numeric_triple": (
        "Answer with the three coordinates in Angstrom, in the order x, y, z, "
        "each to three decimal places.\nExample: FINAL: 12.481, -3.117, 8.226"
    ),
    "distance": (
        "Answer with the distance in Angstrom to two decimal places.\n"
        "Example: FINAL: 3.42"
    ),
    "atom": (
        "Answer with a single atom identifier in the form "
        "<chain>:<one-letter amino-acid code><residue number>:<atom name>.\n"
        "Example: FINAL: A:H57:NE2"
    ),
    "residue": "Answer with a single residue identifier.\nExample: FINAL: A:V22",
    "residue_set": (
        "Answer with a comma-separated list of residue identifiers. Include every "
        "residue that satisfies the criterion and no others. Order does not matter.\n"
        "Example: FINAL: A:D18, A:E21, B:Y44"
    ),
    "residue_pair": (
        "Answer with a residue pair joined by a double hyphen.\n"
        "Example: FINAL: A:C24--B:C81"
    ),
    "residue_pair_set": (
        "Answer with a comma-separated list of residue pairs, each joined by a "
        "double hyphen.\nExample: FINAL: A:C24--A:C79, B:C15--B:C66"
    ),
    "category": "Answer with exactly one of the listed categories.\nExample: FINAL: helix",
    "boolean": "Answer yes or no.\nExample: FINAL: yes",
    "multiple_choice": "Answer with a single option letter.\nExample: FINAL: B",
    "ordered_path": (
        "Answer with the residues in order, separated by ->.\n"
        "Example: FINAL: A:R10 -> A:F42 -> B:E77"
    ),
    "two_interaction_sets": (
        "Report two comma-separated lists of residue pairs, using this exact layout:\n"
        "FINAL\n"
        "gained: A:R10--B:E77, A:Y15--L:L401\n"
        "lost: A:D22--A:K91\n"
        "Write `none` for an empty list."
    ),
}

#: Model-visible question text. ``{...}`` placeholders are filled by the generator.
QUESTION_TEMPLATES = {
    "P01": "List all chain identifiers that appear in the structure.",
    "P02": "How many protein residues are present in chain {chain}?",
    "P03": (
        "Report the coordinates of atom {atom} exactly as they appear in the "
        "structure above."
    ),
    "G01": "What is the distance in Angstrom between atoms {atom1} and {atom2}?",
    "G02": (
        "Which protein heavy atom is closest to atom {atom}, excluding {atom} itself "
        "and any atom connected to it by one or two covalent bonds?"
    ),
    "G03": (
        "Which of the following residues has the smallest minimum heavy-atom distance "
        "to residue {target}?\nCandidates: {candidates}"
    ),
    "G04": (
        "Two residues in this structure are closer together than their atomic radii "
        "allow, producing the single most severe steric clash in the structure. "
        "Which residue pair is it?"
    ),
    "S01": (
        "Residue {residue} forms exactly one salt bridge in this structure. A salt "
        "bridge is present when a Lys NZ or Arg NE/NH1/NH2 nitrogen lies within "
        "4.0 Angstrom of an Asp OD1/OD2 or Glu OE1/OE2 oxygen. Which residue is its "
        "salt-bridge partner?"
    ),
    "S02": (
        "Exactly one protein residue in this structure is phosphorylated, that is, it "
        "is a phosphoserine, phosphothreonine or phosphotyrosine residue. Which "
        "residue is it?"
    ),
    "S03": (
        "Is residue {residue} buried or solvent-exposed? Use the solvent-accessible "
        "surface area of the residue relative to its maximum accessible area: buried "
        "means at most 0.20, solvent-exposed means at least 0.40. Answer buried or "
        "solvent-exposed."
    ),
    "S04": (
        "What is the secondary structure at residue {residue}? Answer helix, strand "
        "or coil."
    ),
    "S05": (
        "Classify chain {chain} as predominantly alpha helical, predominantly beta "
        "sheet, or mixed alpha/beta. Answer with exactly one of: predominantly alpha "
        "helical, predominantly beta sheet, mixed alpha/beta."
    ),
    "S06": (
        "List every protein residue with at least one heavy atom within 4.0 Angstrom "
        "of any heavy atom of {ligand}."
    ),
    "S07": (
        "List every protein residue that directly coordinates the {metal_name} ion "
        "{metal}. A protein nitrogen, oxygen or sulfur atom directly coordinates the "
        "ion when it lies within {cutoff} Angstrom of it. Carbon is never a donor and "
        "water-mediated contacts do not count."
    ),
    "S08": (
        "Cysteine {residue} forms a disulfide bond. Which cysteine is its partner? "
        "A disulfide bond is present when two cysteine SG atoms lie within "
        "2.3 Angstrom of each other."
    ),
    "S09": (
        "Classify the chi1 side-chain rotamer of residue {residue}. Chi1 is the "
        "N-CA-CB-{atom4} dihedral angle, normalised to [-180, 180) degrees. Answer "
        "g+ for 0 <= chi1 < 120, t for chi1 >= 120 or chi1 < -120, and g- for "
        "-120 <= chi1 < 0."
    ),
    "I01": (
        "List every residue in chain {chain_a} that has at least one heavy atom "
        "within 4.0 Angstrom of any heavy atom of chain {chain_b}."
    ),
    "N01": (
        "Exactly one residue contacts both {anchor_a} and {anchor_b}. Two residues "
        "contact when any pair of their heavy atoms lies within 4.0 Angstrom. Which "
        "residue is it?"
    ),
    "T01": (
        "The two structures above are the same protein in two states, already "
        "superposed and rendered in the same coordinate frame. For each candidate "
        "residue pair below, decide whether the contact is gained in Structure 2 "
        "(the residues are in contact in Structure 2 but not in Structure 1) or lost "
        "in Structure 2 (they are in contact in Structure 1 but not in Structure 2). "
        "Residues are in contact when any pair of their heavy atoms lies within "
        "4.0 Angstrom.\nCandidate pairs: {candidates}"
    ),
}

#: Per-family context sentences shown before the coordinates.
CONTEXT_TEMPLATES = {
    "default": "You are given one molecular structure.",
    "two_state": "You are given two molecular structures.",
}

#: Replacements for the context-only control. The generic openers assert that
#: the model has been given a structure, which is precisely what this variant
#: withholds; a prompt that reads as malformed measures confusion rather than
#: the guessing floor it is there to establish. Mechanistic episodes supply
#: their own experimental context and so match nothing here.
CONTEXT_ONLY_SUBSTITUTIONS = {
    CONTEXT_TEMPLATES["default"]: "You are asked about one molecular structure.",
    CONTEXT_TEMPLATES["two_state"]: "You are asked about two molecular structures.",
}


def prompt_fingerprint() -> str:
    """Content hash over every piece of model-visible prompt text."""
    return stable_hash(
        PROMPT_VERSION,
        SYSTEM_PROMPT,
        sorted(REPRESENTATION_NOTICE.items()),
        CROP_NOTICE,
        ROTATION_NOTICE,
        sorted(FORMAT_INSTRUCTIONS.items()),
        sorted(QUESTION_TEMPLATES.items()),
        sorted(CONTEXT_TEMPLATES.items()),
        sorted(CONTEXT_ONLY_SUBSTITUTIONS.items()),
    )[:16]
