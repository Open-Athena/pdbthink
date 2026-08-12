"""Normalized-coordinate rendering: the matched control representation.

The table must contain exactly the same atoms, coordinates, residue labels and
entity labels as the minimal-PDB rendering, at the same three-decimal precision,
so the paired comparison isolates the cost of parsing fixed-column PDB syntax.
"""

from __future__ import annotations

from ..preprocessing.model import Structure

HEADER = "record,chain,resnum,resname,atom,element,x,y,z"


def render_table(structure: Structure, *, decimals: int = 3) -> str:
    fmt = f"%.{decimals}f"
    lines = [HEADER]
    for res in structure.residues:
        record = "ATOM" if res.polymer_kind is not None else "HETATM"
        for atom in res.atoms:
            lines.append(
                ",".join(
                    [
                        record,
                        res.chain,
                        str(res.seq_id),
                        res.name,
                        atom.name,
                        atom.element.upper(),
                        fmt % atom.pos[0],
                        fmt % atom.pos[1],
                        fmt % atom.pos[2],
                    ]
                )
            )
    return "\n".join(lines) + "\n"


def parse_table(text: str) -> list[dict[str, str]]:
    """Inverse of :func:`render_table`, used by equivalence tests."""
    rows: list[dict[str, str]] = []
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines or lines[0] != HEADER:
        raise ValueError("normalized-coordinate table is missing its header")
    columns = HEADER.split(",")
    for line in lines[1:]:
        values = line.split(",")
        if len(values) != len(columns):
            raise ValueError(f"malformed table row: {line!r}")
        rows.append(dict(zip(columns, values)))
    return rows
