"""The six manually curated mechanistic episodes (specification section 11).

Each episode is declared here as data: its source structures, the chains and
non-polymer entities that must survive processing, the model-visible context, the
scored fields and the multiple-choice mechanism options.

The paper-derived claims are *not* here in plaintext: they live encrypted in
``episode_claims.json.gpg`` and are decrypted on demand, because the mechanism
letter cannot be recomputed from coordinates and so a leak of it is permanent.
Residue selections define candidate sets and crop centres. The pipeline
recomputes geometric answers from displayed coordinates under explicit query
rules. Mechanism letters remain curated interpretations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..prompts.library import MECHANISTIC_QUESTIONS
from ..secrets_store import decrypt_json

MECHANISTIC_VERSION = "2.0.0"

#: Curated claims live encrypted so the mechanism letters do not sit in a
#: crawlable file. The passphrase is committed beside them: this is obfuscation
#: against incidental ingestion, not security. See docs/contamination.md.
CLAIMS_PATH = Path(__file__).resolve().parent / "episode_claims.json.gpg"


@lru_cache(maxsize=1)
def load_claims() -> dict[str, dict[str, Any]]:
    """Decrypt the curated episode claims once per process."""
    return decrypt_json(CLAIMS_PATH)


@dataclass
class StateSpec:
    """One state of an episode."""

    entry: str
    chains: list[str]
    keep_components: list[str] = field(default_factory=list)
    #: original component code -> stable model-visible label (A.5)
    ligand_labels: dict[str, str] = field(default_factory=dict)
    #: chain -> author-numbering ranges to retain (drops crystallisation fusions)
    residue_ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    #: chain -> [(first, last, new_chain)] for fused peptides
    split_chains: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    assembly_id: str | None = None
    note: str = ""


@dataclass
class FieldSpec:
    """One deterministically scored answer field."""

    name: str
    prompt: str
    schema: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSpec:
    id: str
    title: str
    state1: StateSpec
    state2: StateSpec
    context: str
    fields: list[FieldSpec]
    mechanism_options: dict[str, str]
    #: state-1 chain -> state-2 chain
    chain_map: dict[str, str] = field(default_factory=dict)
    #: (chain, first, last) state-1 ranges kept out of the alignment core (A.27.3)
    alignment_exclusions: list[tuple[str, int, int]] = field(default_factory=list)
    protein_group_id: str = ""
    notes: str = ""
    #: Episodes that share a source protein for bootstrap clustering (section 11)
    cluster: str = ""
    compute: Callable[..., dict[str, Any]] | None = None

    @property
    def claims(self) -> dict[str, Any]:
        """Curated, paper-derived expectations, decrypted on demand.

        Residue selections become query parameters or verified expectations.
        Mechanism letters remain curated and cannot be recomputed, so a leak
        of them is permanent. See ``docs/contamination.md``.
        """
        return load_claims()[self.id]

    @property
    def entries(self) -> tuple[str, str]:
        return (self.state1.entry, self.state2.entry)

    def mechanism_text(self) -> str:
        return "\n".join(f"{k}. {v}" for k, v in sorted(self.mechanism_options.items()))


MECHANISM_LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# Episode 1: M2 receptor positive allosteric modulation
# --------------------------------------------------------------------------- #

EPISODE_1 = EpisodeSpec(
    id="M1_M2R_PAM",
    title="M2 receptor positive allosteric modulation",
    protein_group_id="m2_receptor",
    cluster="m2_receptor",
    state1=StateSpec(
        entry="4MQS",
        chains=["A"],
        keep_components=["IXO"],
        ligand_labels={"IXO": "L1"},
        note="orthosteric agonist only",
    ),
    state2=StateSpec(
        entry="4MQT",
        chains=["A"],
        keep_components=["IXO", "2CU"],
        ligand_labels={"IXO": "L1", "2CU": "L2"},
        note="orthosteric agonist plus the LY2119620 positive allosteric modulator",
    ),
    context=(
        "Both structures contain the same receptor and orthosteric agonist. One structure "
        "additionally contains an allosteric ligand, L2. Binding experiments show that L2 "
        "increases the receptor's affinity for the orthosteric agonist. Compare the "
        "extracellular ligand-binding vestibules."
    ),
    fields=[
        FieldSpec(
            name="changed_residue",
            prompt=MECHANISTIC_QUESTIONS["ligand_rotamer"],
            schema="residue",
            parameters={"ligand": "L2"},
        ),
        FieldSpec(
            name="gained_interactions",
            prompt=(
                "2. What new local packing arrangement is enabled? Report every pair "
                "consisting of that residue and a partner it contacts in Structure 2 "
                "(within 4.0 Angstrom) but not in Structure 1, as a comma-separated list "
                "of pairs joined by a double hyphen."
            ),
            schema="residue_pair_set",
        ),
        FieldSpec(
            name="mechanism",
            prompt="3. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "L2 stabilizes the agonist-bound extracellular conformation by creating "
            "additional packing in the vestibule."
        ),
        "B": "L2 competes with and ejects the orthosteric agonist.",
        "C": "L2 covalently modifies an intracellular residue.",
        "D": "L2 changes only the global coordinate orientation.",
    },
    chain_map={"A": "A"},
)


# --------------------------------------------------------------------------- #
# Episode 2: CB1 aromatic twin toggle switch
# --------------------------------------------------------------------------- #

EPISODE_2 = EpisodeSpec(
    id="M2_CB1_TOGGLE",
    title="CB1 aromatic twin toggle switch",
    protein_group_id="cb1_receptor",
    cluster="cb1_receptor",
    state1=StateSpec(
        entry="5TGZ",
        chains=["A"],
        keep_components=["ZDG"],
        ligand_labels={"ZDG": "L1"},
        residue_ranges={"A": [(90, 450)]},
        note="antagonist-bound inactive state (AM6538); flavodoxin fusion removed",
    ),
    state2=StateSpec(
        entry="5XRA",
        chains=["A"],
        keep_components=["8D3"],
        ligand_labels={"8D3": "L1"},
        residue_ranges={"A": [(90, 450)]},
        note="agonist-bound state (AM11542); flavodoxin fusion removed",
    ),
    context=(
        "Structure 1 contains a receptor bound to an antagonist that stabilizes an inactive "
        "state. Structure 2 contains the same receptor bound to an agonist. Compare the "
        "ligand-binding pockets and coupled intracellular-facing conformations."
    ),
    fields=[
        FieldSpec(
            name="changed_residues",
            prompt=MECHANISTIC_QUESTIONS["changed_residues"],
            schema="residue_set",
        ),
        FieldSpec(
            name="packing_change",
            prompt=MECHANISTIC_QUESTIONS["packing_change"],
            schema="category",
            parameters={"categories": ["closer", "farther"]},
        ),
        FieldSpec(
            name="mechanism",
            prompt="3. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "Agonist binding reorganizes an aromatic toggle pair, coupling ligand-pocket "
            "contraction to an activation-compatible intracellular surface."
        ),
        "B": "Activation results from breaking a disulfide bond.",
        "C": "The structures differ only by rigid-body rotation.",
        "D": "The antagonist activates the receptor through a covalent aromatic adduct.",
    },
    chain_map={"A": "A"},
    alignment_exclusions=[("A", 195, 210), ("A", 350, 362)],
)


# --------------------------------------------------------------------------- #
# Episode 3: A2A receptor activation by G-protein engagement
# --------------------------------------------------------------------------- #

EPISODE_3 = EpisodeSpec(
    id="M3_A2A_GPROTEIN",
    title="A2A receptor activation by G-protein engagement",
    protein_group_id="a2a_receptor",
    cluster="a2a_receptor",
    state1=StateSpec(
        entry="3QAK",
        chains=["A"],
        keep_components=["UKA"],
        ligand_labels={"UKA": "L1"},
        residue_ranges={"A": [(1, 320)]},
        note="active-intermediate receptor with the UK-432097 agonist; T4 lysozyme fusion removed",
    ),
    state2=StateSpec(
        entry="5G53",
        chains=["A", "C"],
        keep_components=["NEC"],
        ligand_labels={"NEC": "L1"},
        note="receptor chain A with the NECA agonist and mini-Gs chain C",
    ),
    context=(
        "Both structures contain an agonist-bound receptor. One is an active-intermediate "
        "state without a G protein; the other is a G-protein-bound active state. The "
        "deposited agonists differ, so restrict your analysis to the cytoplasmic half of the "
        "receptor."
    ),
    fields=[
        FieldSpec(
            name="changed_residues",
            prompt=MECHANISTIC_QUESTIONS["changed_residues"],
            schema="residue_set",
        ),
        FieldSpec(
            name="helix6_displacement",
            prompt=MECHANISTIC_QUESTIONS["helix6_displacement"],
            schema="distance",
            parameters={"tolerance": 2.0},
        ),
        FieldSpec(
            name="new_contact",
            prompt=MECHANISTIC_QUESTIONS["new_contact"],
            schema="residue_pair",
        ),
        FieldSpec(
            name="mechanism",
            prompt="4. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "G-protein engagement stabilizes outward movement of helix 6, repacking conserved "
            "side chains and creating a receptor-G-protein interface."
        ),
        "B": "It restructures only the extracellular agonist pocket.",
        "C": "The G protein replaces the orthosteric ligand.",
        "D": "Binding causes no receptor conformational change.",
    },
    chain_map={"A": "A"},
    alignment_exclusions=[("A", 222, 240)],
    notes="The cytoplasmic end of helix 6 is excluded from the alignment core (A.27.3).",
)


# --------------------------------------------------------------------------- #
# Episode 4: Beta2AR-Gs interface exchange
# --------------------------------------------------------------------------- #

EPISODE_4 = EpisodeSpec(
    id="M4_B2AR_ALPHA5",
    title="Beta2AR-Gs alpha5 interface exchange",
    protein_group_id="b2ar_gs",
    cluster="b2ar_gs",
    state1=StateSpec(
        entry="6E67",
        chains=["A"],
        keep_components=[],
        residue_ranges={"A": [(29, 343), (2381, 2394)]},
        split_chains={"A": [(2381, 2394, "P")]},
        note=(
            "receptor plus the C-terminal alpha5 peptide of G-alpha-s, which is fused into "
            "the receptor chain at 2381-2394 and is moved to its own chain P so that residue "
            "mapping sees only the receptor"
        ),
    ),
    state2=StateSpec(
        entry="3SN6",
        chains=["R", "A"],
        keep_components=[],
        residue_ranges={"R": [(29, 343)], "A": [(1, 400)]},
        note=(
            "mature nucleotide-free complex: receptor chain R (T4 lysozyme fusion removed) "
            "and G-alpha-s chain A"
        ),
    ),
    context=(
        "Structure 1 represents an initial receptor interaction with the C-terminal alpha5 "
        "region of a G protein. Structure 2 represents the mature nucleotide-free "
        "receptor-G-protein complex. Compare which face of the alpha5 helix contacts the "
        "receptor."
    ),
    fields=[
        FieldSpec(
            name="initial_residues",
            prompt=MECHANISTIC_QUESTIONS["initial_residues"],
            schema="residue_set",
        ),
        FieldSpec(
            name="mature_residues",
            prompt=MECHANISTIC_QUESTIONS["mature_residues"],
            schema="residue_set",
        ),
        FieldSpec(
            name="mechanism",
            prompt="3. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "Alpha5 reorients during maturation, replacing the initial contacts with another "
            "receptor-facing surface and coupling engagement to G-protein rearrangement."
        ),
        "B": "The alpha5 interface is identical in both states.",
        "C": "G-beta replaces alpha5 at the interface.",
        "D": "The states differ only by coordinate orientation.",
    },
    chain_map={"A": "R"},
    notes=(
        "The fused alpha5 peptide and full G-alpha-s are matched by sequence, not by "
        "assuming identical PDB chain identifiers. 6EG8 is deliberately not used as the "
        "mature comparator because it is receptor-free GDP-bound Gs."
    ),
)


# --------------------------------------------------------------------------- #
# Episodes 5 and 6: TRPV1
# --------------------------------------------------------------------------- #

EPISODE_5 = EpisodeSpec(
    id="M5_TRPV1_Y511",
    title="TRPV1 Tyr511 rotamer change after lipid displacement",
    protein_group_id="trpv1",
    cluster="trpv1",
    state1=StateSpec(
        entry="5IRZ",
        chains=["B"],
        keep_components=["6ES", "6O8", "6OE"],
        ligand_labels={"6ES": "L2", "6O8": "L3", "6OE": "L4"},
        note="unliganded protomer in a lipid nanodisc, resident lipids retained",
    ),
    state2=StateSpec(
        entry="5IRX",
        chains=["A"],
        keep_components=["6EU", "6O8", "6OE"],
        ligand_labels={"6EU": "L1", "6O8": "L3", "6OE": "L4"},
        note="resiniferatoxin-bound protomer",
    ),
    context=(
        "Structure 1 is an unliganded ion channel in a lipid environment. Structure 2 "
        "contains an activating ligand in a transmembrane pocket. Experiments show that "
        "occupancy of this pocket promotes opening. Compare the local pocket geometry."
    ),
    fields=[
        FieldSpec(
            name="changed_residue",
            prompt=(
                "1. Which aromatic residue undergoes the clearest ligand-associated rotamer "
                "change? Report one residue identifier."
            ),
            schema="residue",
        ),
        FieldSpec(
            name="displaced_entity",
            prompt=(
                "2. What occupies the relevant space in Structure 1? Report the identifier of "
                "the non-polymer entity."
            ),
            schema="residue",
        ),
        FieldSpec(
            name="gained_interaction",
            prompt=(
                "3. What interaction becomes possible after displacement? Report one residue "
                "pair joined by a double hyphen."
            ),
            schema="residue_pair",
        ),
        FieldSpec(
            name="mechanism",
            prompt="4. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "The activating ligand displaces a resident lipid, allowing an aromatic side chain "
            "to reorient and contact the ligand."
        ),
        "B": "The activating ligand preserves the unliganded lipid pose and prevents rearrangement.",
        "C": "Activation occurs by cleavage of the aromatic residue.",
        "D": "The ligand binds only to the extracellular toxin site.",
    },
    chain_map={"B": "A"},
    alignment_exclusions=[("B", 505, 520)],
    notes="Score hydrogen-bond identity only when ligand donor/acceptor chemistry is represented.",
)


EPISODE_6 = EpisodeSpec(
    id="M6_TRPV1_R557_E570",
    title="TRPV1 Arg557-Glu570 gate coupling",
    protein_group_id="trpv1",
    cluster="trpv1",
    state1=StateSpec(
        entry="5IRZ",
        chains=["B", "C"],
        keep_components=["6ES"],
        ligand_labels={"6ES": "L2"},
        note="unliganded state, two adjacent protomers",
    ),
    state2=StateSpec(
        entry="5IRX",
        chains=["A", "B"],
        keep_components=["6EU"],
        ligand_labels={"6EU": "L1"},
        note="agonist-bound state, two adjacent protomers",
    ),
    context=(
        "The structures show unliganded and ligand-bound states of the same ion channel. One "
        "ligand is an agonist that promotes opening. Identify the interaction that most "
        "directly couples the ligand pocket to movement of the intracellular gate."
    ),
    fields=[
        FieldSpec(
            name="gained_pair",
            prompt=MECHANISTIC_QUESTIONS["gained_pair"],
            schema="residue_pair",
        ),
        FieldSpec(
            name="within_protomer",
            prompt=(
                "2. Is the interaction within one protomer (one chain) or between protomers "
                "(two chains)? Answer yes if it is within one chain, no otherwise."
            ),
            schema="boolean",
        ),
        FieldSpec(
            name="mechanism",
            prompt="3. Which explanation is best supported?",
            schema="multiple_choice",
        ),
    ],
    mechanism_options={
        "A": (
            "A new Arg-Glu interaction couples pocket rearrangement to displacement of the "
            "S4-S5 linker and lower-gate opening."
        ),
        "B": "Opening results from breaking an extracellular disulfide.",
        "C": "The charged pair is unchanged and unrelated to gating.",
        "D": "Only the antagonist forms the activating pair.",
    },
    chain_map={"B": "A", "C": "B"},
    alignment_exclusions=[("B", 550, 580), ("C", 550, 580)],
    notes="Shares processed TRPV1 structures with episode 5; both cluster as one protein.",
)


EPISODES: dict[str, EpisodeSpec] = {
    e.id: e for e in (EPISODE_1, EPISODE_2, EPISODE_3, EPISODE_4, EPISODE_5, EPISODE_6)
}

#: PDB entries each episode depends on, used to build acquisition manifests.
EPISODE_SOURCES: dict[str, tuple[str, ...]] = {
    episode_id: episode.entries for episode_id, episode in EPISODES.items()
}
