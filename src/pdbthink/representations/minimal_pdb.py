"""Minimal PDB rendering: the primary model-visible representation (section 6).

Only ``ATOM``, ``HETATM``, ``TER`` and ``END`` records are emitted. Everything
that could leak provenance or pre-computed annotation (headers, titles,
``HELIX``/``SHEET``, ``LINK``, ``SSBOND``, ``ANISOU``, B-factors, pLDDT) is
already gone by the time a structure reaches this module.
"""

from __future__ import annotations

from ..preprocessing.model import EntityType, Structure

COORD_FORMAT = "%8.3f"


def render_minimal_pdb(structure: Structure, *, decimals: int = 3) -> str:
    """Fixed-column PDB text for a displayed structure."""
    lines: list[str] = []
    serial = 1
    previous_chain: str | None = None
    previous_polymer: bool = False
    last_polymer_residue = None

    for res in structure.residues:
        is_polymer = res.polymer_kind is not None
        if previous_chain is not None and (res.chain != previous_chain) and previous_polymer:
            lines.append(_ter_line(serial, last_polymer_residue))
            serial += 1
        for atom in res.atoms:
            lines.append(
                _atom_line(
                    serial=serial,
                    record="ATOM" if is_polymer else "HETATM",
                    atom_name=atom.name,
                    res_name=res.name,
                    chain=res.chain,
                    seq_id=res.seq_id,
                    icode=res.icode,
                    x=atom.pos[0],
                    y=atom.pos[1],
                    z=atom.pos[2],
                    occupancy=atom.occupancy,
                    bfactor=atom.bfactor,
                    element=atom.element,
                    decimals=decimals,
                )
            )
            serial += 1
        previous_chain = res.chain
        previous_polymer = is_polymer
        if is_polymer:
            last_polymer_residue = res

    if previous_polymer and last_polymer_residue is not None:
        lines.append(_ter_line(serial, last_polymer_residue))
    lines.append("END")
    return "\n".join(lines) + "\n"


def _atom_line(
    *,
    serial: int,
    record: str,
    atom_name: str,
    res_name: str,
    chain: str,
    seq_id: int,
    icode: str,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
    element: str,
    decimals: int,
) -> str:
    name = _align_atom_name(atom_name, element)
    coord = f"%8.{decimals}f"
    return (
        f"{record:<6}{serial:>5} {name}{' ':1}{res_name:>3} {chain:1}{seq_id:>4}{icode:1}   "
        f"{coord % x}{coord % y}{coord % z}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element.upper():>2}  "
    ).rstrip()


def _align_atom_name(name: str, element: str) -> str:
    """PDB atom-name alignment: one-character elements start in column 14."""
    name = name.strip()
    if len(name) >= 4:
        return name[:4]
    if len(element.strip()) == 1 and not name[:1].isdigit():
        return f" {name:<3}"
    return f"{name:<4}"


def _ter_line(serial: int, res) -> str:
    return f"TER   {serial:>5}      {res.name:>3} {res.chain:1}{res.seq_id:>4}{res.icode:1}".rstrip()


def render_entity_legend(structure: Structure) -> str:
    """Human-readable legend for non-polymer entities in the rendering.

    Anonymised ligands need a model-visible label so a question can refer to
    them without revealing the source component code (A.5).
    """
    entries: list[str] = []
    for res in structure.residues:
        if res.entity is EntityType.LIGAND:
            entries.append(f"{res.label}: non-polymer ligand with {len(res.atoms)} heavy atoms")
        elif res.entity is EntityType.METAL:
            entries.append(f"{res.label}: {res.element_symbol.capitalize()} ion")
    return "\n".join(entries)
