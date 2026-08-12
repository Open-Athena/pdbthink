"""Generator framework.

Every question family is an independent generator with two responsibilities:

``propose``
    Search a processed structure for question parameters that satisfy the
    Appendix A ambiguity margins, and record why each candidate was accepted or
    rejected.

``oracle``
    Recompute the gold answer and its hidden evidence from *any* displayed
    structure given those parameters.

Splitting the two is what makes the guarantees in "definition of done"
mechanical: gold answers are always computed from the coordinates the model
sees, rotation variants and paired representations are checked by running the
same oracle again, and a crop is accepted only when the oracle still returns the
same answer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np

from ..config import Definitions, ProteinSpec
from ..geometry.contacts import (
    ContactGraph,
    build_contact_graph,
    find_clashes,
    find_disulfides,
    residue_contacts,
    salt_bridges,
)
from ..geometry.dssp import DsspResult, assign_dssp
from ..geometry.sasa import SasaResult, compute_sasa
from ..geometry.topology import Topology, build_topology
from ..preprocessing.loader import ProcessedStructure
from ..preprocessing.model import Structure
from ..preprocessing.transform import DisplayedStructure

#: Coarse size guard for families that may not be cropped. A rendered minimal-PDB
#: atom line costs about 40 cl100k tokens (measured across the V1 sources), so
#: this keeps whole-structure prompts inside the 64,000-token automatic budget
#: before the builder's exact per-render check.
MAX_UNCROPPABLE_ATOMS = 1560


class Analysis:
    """Lazily computed, memoised geometry for one displayed structure."""

    def __init__(self, structure: Structure, definitions: Definitions) -> None:
        self.structure = structure
        self.definitions = definitions

    @cached_property
    def disulfides(self):
        return find_disulfides(self.structure, self.definitions)

    @cached_property
    def topology(self) -> Topology:
        return build_topology(self.structure, [(s.i, s.j) for s in self.disulfides])

    @cached_property
    def dssp(self) -> DsspResult:
        return assign_dssp(self.structure, self.definitions)

    @cached_property
    def sasa(self) -> SasaResult:
        return compute_sasa(self.structure, self.definitions)

    @cached_property
    def contacts(self):
        return residue_contacts(self.structure, self.definitions)

    @cached_property
    def graph(self) -> ContactGraph:
        return build_contact_graph(self.structure, self.definitions)

    @cached_property
    def salt_bridges(self):
        return salt_bridges(self.structure, self.definitions)

    @cached_property
    def clashes(self):
        return find_clashes(self.structure, self.topology, self.definitions)

    @cached_property
    def secondary_runs(self) -> dict[int, tuple[int, int]]:
        """Per residue: (length of its three-state run, distance to run edge)."""
        out: dict[int, tuple[int, int]] = {}
        for chain in self.structure.protein_chains:
            ordered = sorted(
                [
                    i
                    for i, r in enumerate(self.structure.residues)
                    if r.is_protein and r.chain == chain
                ],
                key=lambda i: self.structure.residues[i].poly_index or 0,
            )
            run: list[int] = []
            current: str | None = None
            for ri in ordered:
                klass = self.dssp.three_state.get(ri)
                if klass is None:
                    klass = "__unassigned__"
                if klass != current:
                    _flush_run(out, run)
                    run, current = [ri], klass
                else:
                    run.append(ri)
            _flush_run(out, run)
        return out


def _flush_run(out: dict[int, tuple[int, int]], run: list[int]) -> None:
    for position, ri in enumerate(run):
        out[ri] = (len(run), min(position, len(run) - 1 - position))


@dataclass
class GenerationContext:
    """Everything a generator needs for one source structure."""

    spec: ProteinSpec
    processed: ProcessedStructure
    displayed: DisplayedStructure
    definitions: Definitions
    rng: np.random.Generator
    #: Shared across families so DSSP and SASA are computed once per structure.
    analysis: Analysis | None = None

    def __post_init__(self) -> None:
        if self.analysis is None:
            self.analysis = Analysis(self.structure, self.definitions)

    @property
    def structure(self) -> Structure:
        return self.displayed.structure


@dataclass
class Proposal:
    """A candidate question that passed its generator's margin checks."""

    parameters: dict[str, Any]
    margins: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    criteria_passed: list[str] = field(default_factory=list)
    required_labels: list[str] = field(default_factory=list)
    crop_centers: list[str] = field(default_factory=list)
    rank: float = 0.0                      # lower sorts first; deterministic tie-break
    tag: str = ""                          # diversity key within a family

    def key(self) -> str:
        import json

        return json.dumps(self.parameters, sort_keys=True, default=str)


@dataclass
class Rejection:
    """A candidate that failed a criterion, kept for the rejection log (A.34)."""

    reason: str
    detail: dict[str, Any] = field(default_factory=dict)
    criteria_failed: list[str] = field(default_factory=list)


@dataclass
class OracleResult:
    gold_answer: dict[str, Any]
    evidence: dict[str, Any] = field(default_factory=dict)


class Generator(ABC):
    """Base class for every question family."""

    family: str = ""
    version: str = "1.0.0"
    answer_schema: str = ""
    level: str = ""
    croppable: bool = False
    crop_radius: float = 15.0
    needs_ligand: bool = False
    needs_metal: bool = False
    two_state: bool = False
    #: Extra prompt instructions appended after the question text.
    extra_instructions: str = ""

    @abstractmethod
    def propose(self, ctx: GenerationContext) -> Iterator[Proposal | Rejection]:
        """Yield candidate parameter sets, plus rejections for the log."""

    @abstractmethod
    def oracle(
        self,
        structure: Structure,
        parameters: dict[str, Any],
        definitions: Definitions,
        analysis: Analysis,
    ) -> OracleResult:
        """Recompute the gold answer from a displayed structure."""

    @abstractmethod
    def question(self, parameters: dict[str, Any], structure: Structure) -> str:
        """Model-visible question text."""

    def context(self, parameters: dict[str, Any]) -> str:
        return "You are given one molecular structure."

    def prompt_parameters(self, parameters: dict[str, Any]) -> dict[str, Any]:
        """Extra parameters the scorer needs (categories, options, tolerance)."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.family} v{self.version}>"


_REGISTRY: dict[str, Generator] = {}


def register(generator: Generator) -> Generator:
    if generator.family in _REGISTRY:
        raise ValueError(f"duplicate generator for family {generator.family}")
    _REGISTRY[generator.family] = generator
    return generator


def get_generator(family: str) -> Generator:
    if family not in _REGISTRY:
        raise KeyError(f"no generator registered for family {family!r}")
    return _REGISTRY[family]


def all_generators() -> dict[str, Generator]:
    return dict(sorted(_REGISTRY.items()))


def label_of(structure: Structure, index: int) -> str:
    return structure.residues[index].label
