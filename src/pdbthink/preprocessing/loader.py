"""Load a cached mmCIF into the internal model, applying A.1-A.3 and A.6.

The result of :func:`load_processed` is *not yet* model-visible: it is the
sanitised, altloc-resolved, anonymised structure in the deposited frame. Rigid
transformation, cropping and coordinate rounding happen per rendered variant in
:mod:`pdbthink.preprocessing.transform`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gemmi
import numpy as np

from ..acquisition.cache import SourceRecord
from ..chem import MONATOMIC_METALS, is_amino_acid
from ..config import Definitions, ProteinSpec
from .model import Atom, EntityType, Residue, Structure, polymer_kind_for


class StructureRejected(RuntimeError):
    """Raised when a source structure cannot be used under the V1 definitions."""

    def __init__(self, reason: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class ProcessedStructure:
    """A sanitised structure plus the private provenance the curator sees."""

    structure: Structure
    spec: ProteinSpec
    record: SourceRecord
    assembly_id: str | None
    ligand_map: dict[str, str] = field(default_factory=dict)   # displayed -> original
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    #: Per-residue AlphaFold pLDDT, captured before B-factors are zeroed (A.2.9,
    #: section 9). Private: never model-visible.
    plddt: dict[str, float] = field(default_factory=dict)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "protein_group_id": self.spec.id,
            "source_type": self.spec.source_type,
            "source_entry": self.record.entry,
            "source_file_sha256": self.record.sha256,
            "release_date": self.record.release_date,
            "experimental_method": self.record.experimental_method,
            "resolution": self.record.resolution,
            "title": self.record.title,
            "publications": self.record.publications,
            "assembly_id": self.assembly_id,
            "ligand_map": self.ligand_map,
            "warnings": self.warnings,
        }


def load_processed(
    record: SourceRecord,
    spec: ProteinSpec,
    definitions: Definitions,
    *,
    keep_components: list[str] | None = None,
) -> ProcessedStructure:
    """Parse, filter and anonymise one source structure."""
    sp = definitions.get("structure_processing")
    method = (record.experimental_method or "").upper()
    for banned in sp["exclude_experimental_methods"]:
        if banned.upper() in method:
            raise StructureRejected(
                "excluded_experimental_method", {"method": record.experimental_method}
            )

    st = gemmi.read_structure(record.path)
    st.setup_entities()
    if len(st) == 0:
        raise StructureRejected("no_coordinate_models", {"entry": record.entry})

    assembly_id = spec.assembly_id
    if assembly_id is not None:
        names = [a.name for a in st.assemblies]
        if assembly_id not in names:
            raise StructureRejected(
                "unknown_biological_assembly", {"requested": assembly_id, "available": names}
            )
        # `Short` keeps single-character author chain names where it can, which
        # is what the PDB fixed-column chain field allows.
        st.transform_to_assembly(assembly_id, gemmi.HowToNameCopiedChain.Short)
        st.setup_entities()

    model = st[int(sp["model_index"])]

    discard = {c.upper() for c in sp["discard_components"]}
    selected = {c.upper() for c in (keep_components if keep_components is not None else spec.keep_components)}
    discard -= selected
    wanted_chains = set(spec.chains) if spec.chains else None

    residues: list[Residue] = []
    warnings: list[str] = []
    n_altloc_resolved = 0
    n_dropped_components: dict[str, int] = {}

    for chain in model:
        if wanted_chains is not None and chain.name not in wanted_chains:
            continue
        for res in chain:
            name = res.name.strip().upper()
            info = gemmi.find_tabulated_residue(res.name)
            if (info is not None and info.is_water()) or name in ("HOH", "DOD"):
                continue
            aa = is_amino_acid(name) or (info is not None and info.is_amino_acid())
            nucleic = info is not None and info.is_nucleic_acid()
            polymer = aa or nucleic
            if not polymer:
                if name in discard and name not in selected:
                    n_dropped_components[name] = n_dropped_components.get(name, 0) + 1
                    continue
                if selected and name not in selected:
                    n_dropped_components[name] = n_dropped_components.get(name, 0) + 1
                    continue

            icode = res.seqid.icode or " "
            if polymer and icode.strip() and sp["exclude_insertion_codes"]:
                raise StructureRejected(
                    "insertion_code_present",
                    {"chain": chain.name, "seq_id": res.seqid.num, "icode": icode},
                )

            atoms, resolved = _resolve_altlocs(res, definitions)
            n_altloc_resolved += int(resolved)
            atoms = [a for a in atoms if not a.is_hydrogen] if sp["remove_hydrogens"] else atoms
            if not atoms:
                continue

            entity = _entity_type(name, aa, nucleic, atoms)
            residues.append(
                Residue(
                    chain=chain.name,
                    seq_id=res.seqid.num,
                    name=name,
                    entity=entity,
                    atoms=atoms,
                    icode=icode,
                    orig_name=name,
                    polymer_kind=polymer_kind_for(name) if polymer else None,
                )
            )

    if not residues:
        raise StructureRejected("no_retained_residues", {"entry": record.entry})

    residues = _apply_residue_ranges(residues, spec)
    _apply_chain_splits(residues, spec)
    if not residues:
        raise StructureRejected("residue_ranges_retained_nothing", {"entry": record.entry})

    structure = Structure(residues)
    chain_map = _enforce_single_character_chains(structure)
    if chain_map:
        warnings.append(f"chain identifiers remapped for PDB compatibility: {chain_map}")
    structure.assign_polymer_indices()
    _check_label_collisions(structure)

    ligand_map = _anonymise_ligands(structure)
    plddt = _capture_plddt(structure) if spec.source_type == "afdb" else {}
    _normalise_atom_records(structure, sp)

    if wanted_chains is not None:
        missing = wanted_chains - set(structure.chains)
        if missing:
            raise StructureRejected(
                "requested_chain_absent", {"missing": sorted(missing), "present": structure.chains}
            )

    structure.meta = {
        "source_type": spec.source_type,
        "entry": record.entry,
        "assembly_id": assembly_id,
        "protein_group_id": spec.id,
    }
    stats = {
        "residues": len(structure.residues),
        "protein_residues": len(structure.protein_residues),
        "atoms": structure.atom_count,
        "chains": structure.chains,
        "altloc_residues_resolved": n_altloc_resolved,
        "dropped_components": n_dropped_components,
    }
    return ProcessedStructure(
        structure=structure,
        spec=spec,
        record=record,
        assembly_id=assembly_id,
        ligand_map=ligand_map,
        warnings=warnings,
        stats=stats,
        plddt=plddt,
    )


def _capture_plddt(structure: Structure) -> dict[str, float]:
    """Record AlphaFold per-residue confidence before B-factors are zeroed."""
    out: dict[str, float] = {}
    for res in structure.residues:
        if res.atoms:
            out[res.label] = sum(a.bfactor for a in res.atoms) / len(res.atoms)
    return out


# --------------------------------------------------------------------------- #
# A.3 alternate locations and occupancy
# --------------------------------------------------------------------------- #

def _resolve_altlocs(res: gemmi.Residue, definitions: Definitions) -> tuple[list[Atom], bool]:
    """Pick one conformer per residue; never mix alternate locations per atom."""
    cfg = definitions.get("altloc")
    shared: list[gemmi.Atom] = []
    by_altloc: dict[str, list[gemmi.Atom]] = {}
    for atom in res:
        if cfg["exclude_zero_occupancy"] and atom.occ == 0.0:
            continue
        # gemmi reports a blank alternate location as NUL rather than a space,
        # and NUL is not whitespace, so it has to be removed explicitly. Getting
        # this wrong makes every shared atom look like its own conformer.
        alt = (atom.altloc or "").replace("\x00", "").strip()
        if not alt:
            shared.append(atom)
        else:
            by_altloc.setdefault(alt, []).append(atom)

    chosen_alt: str | None = None
    if by_altloc:
        def summed_occupancy(alt: str) -> float:
            group = by_altloc[alt]
            side = [a for a in group if a.name not in ("N", "CA", "C", "O", "OXT")]
            return sum(a.occ for a in (side or group))

        # Highest summed side-chain occupancy; ties broken alphabetically (A.3.4).
        chosen_alt = min(by_altloc, key=lambda alt: (-summed_occupancy(alt), alt))

    keep = list(shared) + (by_altloc[chosen_alt] if chosen_alt else [])
    seen: set[str] = set()
    out: list[Atom] = []
    for a in keep:
        if a.name in seen:
            continue          # microheterogeneity residue: keep first, flagged below
        seen.add(a.name)
        out.append(
            Atom(
                name=a.name.strip(),
                element=a.element.name.upper(),
                pos=np.array([a.pos.x, a.pos.y, a.pos.z], dtype=float),
                altloc="",
                occupancy=float(a.occ),
                bfactor=float(a.b_iso),
                serial=int(a.serial),
                is_hetatm=res.het_flag == "H",
            )
        )
    return out, chosen_alt is not None


def _entity_type(name: str, aa: bool, nucleic: bool, atoms: list[Atom]) -> EntityType:
    if aa:
        return EntityType.PROTEIN
    if nucleic:
        return EntityType.NUCLEIC
    if name in MONATOMIC_METALS and len(atoms) == 1:
        return EntityType.METAL
    return EntityType.LIGAND


def _apply_residue_ranges(residues: list[Residue], spec: ProteinSpec) -> list[Residue]:
    """Drop residues outside the configured author-numbering ranges."""
    if not spec.residue_ranges:
        return residues
    out: list[Residue] = []
    for res in residues:
        ranges = spec.residue_ranges.get(res.chain)
        if ranges is None:
            out.append(res)
            continue
        if any(first <= res.seq_id <= last for first, last in ranges):
            out.append(res)
    return out


def _apply_chain_splits(residues: list[Residue], spec: ProteinSpec) -> None:
    """Move a numbered range into its own chain (fused peptides, A.27)."""
    if not spec.split_chains:
        return
    for res in residues:
        for first, last, new_chain in spec.split_chains.get(res.chain, []):
            if first <= res.seq_id <= last:
                res.chain = new_chain
                break


_CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def _enforce_single_character_chains(structure: Structure) -> dict[str, str]:
    """Guarantee chain IDs fit the one-column PDB chain field.

    Author chain names are kept whenever they are already single characters and
    unique (A.5). Assembly expansion can produce longer names, in which case the
    whole set is remapped deterministically in order of appearance and the map is
    recorded as private provenance.
    """
    names = structure.chains
    if all(len(n) == 1 for n in names):
        return {}
    used = {n for n in names if len(n) == 1}
    mapping: dict[str, str] = {}
    for name in names:
        if len(name) == 1:
            mapping[name] = name
            continue
        for candidate in _CHAIN_ALPHABET:
            if candidate not in used:
                used.add(candidate)
                mapping[name] = candidate
                break
        else:  # pragma: no cover - more than 62 chains
            raise StructureRejected("too_many_chains", {"chains": names})
    for res in structure.residues:
        res.chain = mapping[res.chain]
    structure.invalidate()
    return {k: v for k, v in mapping.items() if k != v}


def _check_label_collisions(structure: Structure) -> None:
    seen: dict[str, Residue] = {}
    for res in structure.residues:
        if res.label in seen:
            other = seen[res.label]
            raise StructureRejected(
                "ambiguous_residue_label",
                {"label": res.label, "first": str(other.key), "second": str(res.key)},
            )
        seen[res.label] = res


def _anonymise_ligands(structure: Structure) -> dict[str, str]:
    """Replace non-metal ligand component codes with stable ``L1..Ln`` labels.

    Metals keep their element code (needed to answer coordination questions) and
    modified polymer residues keep their component code (chemistry, not
    provenance). Every copy of the same original component gets the same label.
    """
    mapping: dict[str, str] = {}
    next_index = 1
    for res in structure.residues:
        if res.entity is not EntityType.LIGAND:
            continue
        if res.orig_name not in mapping:
            mapping[res.orig_name] = f"L{next_index}"
            next_index += 1
        res.name = mapping[res.orig_name]
    structure.invalidate()
    return {v: k for k, v in mapping.items()}


def _normalise_atom_records(structure: Structure, sp: dict[str, Any]) -> None:
    """A.2 steps 7-9: renumber serials, uniform occupancy, zeroed B-factors."""
    serial = 1
    for res in structure.residues:
        for atom in res.atoms:
            atom.serial = serial
            atom.occupancy = float(sp["occupancy_value"])
            atom.bfactor = float(sp["bfactor_value"])
            serial += 1
    structure.invalidate()
