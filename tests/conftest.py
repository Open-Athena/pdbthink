"""Shared fixtures.

Every test runs offline: the committed structures under ``tests/fixtures`` are
already sanitised minimal-PDB renderings, so they are loaded through the same
:func:`~pdbthink.preprocessing.load_processed` path as cached mmCIF sources but
need no network access.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pdbthink.acquisition.cache import SourceRecord
from pdbthink.config import Definitions, ProteinSpec
from pdbthink.preprocessing import load_processed
from pdbthink.preprocessing.model import Atom, EntityType, Residue, Structure

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def definitions() -> Definitions:
    return Definitions.load()


def fixture_record(name: str, entry: str) -> SourceRecord:
    return record_for_path(FIXTURES / name, entry)


def record_for_path(path: Path, entry: str) -> SourceRecord:
    """A minimal provenance record pointing at a local file."""
    return SourceRecord(
        key=f"fixture:{entry}",
        source_type="pdb",
        entry=entry,
        path=str(path),
        sha256="0" * 64,
        bytes=path.stat().st_size,
        url="file://fixture",
        release_date="1981-07-28",
        experimental_method="X-RAY DIFFRACTION",
        resolution=1.5,
    )


def load_fixture(name: str, entry: str, definitions: Definitions, **spec_kwargs):
    spec = ProteinSpec(id=entry.lower(), source_type="pdb", entry=entry, **spec_kwargs)
    return load_processed(fixture_record(name, entry), spec, definitions)


@pytest.fixture(scope="session")
def crambin(definitions):
    return load_fixture("crambin_processed.pdb", "1CRN", definitions)


@pytest.fixture(scope="session")
def zinc_site(definitions):
    return load_fixture("zinc_site.pdb", "1CA2", definitions, keep_components=["ZN"])


def make_residue(
    chain: str,
    seq_id: int,
    name: str,
    atoms: dict[str, tuple[float, float, float]],
    entity: EntityType = EntityType.PROTEIN,
    polymer_kind: str | None = "protein",
) -> Residue:
    return Residue(
        chain=chain,
        seq_id=seq_id,
        name=name,
        entity=entity,
        polymer_kind=polymer_kind,
        atoms=[
            Atom(name=n, element=n[0], pos=np.array(p, dtype=float))
            for n, p in atoms.items()
        ],
    )


def make_structure(residues: list[Residue]) -> Structure:
    structure = Structure(residues)
    structure.assign_polymer_indices()
    return structure


def place_atom(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, bond: float, angle: float, torsion: float
) -> np.ndarray:
    """Natural-extension reference frame: place D from A-B-C plus internal coordinates."""
    angle, torsion = math.radians(angle), math.radians(torsion)
    bc = c - b
    bc /= np.linalg.norm(bc)
    n = np.cross(b - a, bc)
    n /= np.linalg.norm(n)
    m = np.cross(n, bc)
    local = np.array(
        [
            -bond * math.cos(angle),
            bond * math.sin(angle) * math.cos(torsion),
            bond * math.sin(angle) * math.sin(torsion),
        ]
    )
    return c + local[0] * bc + local[1] * m + local[2] * n


def build_backbone(
    phi: float, psi: float, n_residues: int, chain: str = "A", omega: float = 180.0
) -> Structure:
    """Backbone built from standard internal coordinates at fixed phi/psi.

    Using real bond lengths and angles rather than a hand-drawn spiral matters:
    the Kabsch-Sander hydrogen-bond energy is sensitive to the N-H...O=C geometry,
    so only a properly constructed backbone reproduces textbook DSSP assignments.
    """
    n_ca, ca_c, c_n = 1.458, 1.525, 1.329
    a_n_ca_c, a_ca_c_n, a_c_n_ca = 111.2, 116.2, 121.7
    c_o, a_ca_c_o = 1.231, 120.8

    n0 = np.array([0.0, 0.0, 0.0])
    ca0 = np.array([n_ca, 0.0, 0.0])
    c0 = place_atom(np.array([0.0, 1.0, 0.0]), n0, ca0, ca_c, a_n_ca_c, 0.0)
    backbone = [{"N": n0, "CA": ca0, "C": c0}]

    for _ in range(1, n_residues):
        previous = backbone[-1]
        n = place_atom(previous["N"], previous["CA"], previous["C"], c_n, a_ca_c_n, psi)
        ca = place_atom(previous["CA"], previous["C"], n, n_ca, a_c_n_ca, omega)
        c = place_atom(previous["C"], n, ca, ca_c, a_n_ca_c, phi)
        backbone.append({"N": n, "CA": ca, "C": c})

    residues: list[Residue] = []
    for i, atoms in enumerate(backbone):
        following = backbone[i + 1]["N"] if i + 1 < len(backbone) else None
        if following is None:
            oxygen = place_atom(atoms["N"], atoms["CA"], atoms["C"], c_o, a_ca_c_o, psi - 180.0)
        else:
            oxygen = place_atom(following, atoms["CA"], atoms["C"], c_o, a_ca_c_o, 180.0)
        residues.append(
            make_residue(
                chain,
                i + 1,
                "ALA",
                {
                    "N": tuple(atoms["N"]),
                    "CA": tuple(atoms["CA"]),
                    "C": tuple(atoms["C"]),
                    "O": tuple(oxygen),
                },
            )
        )
    return make_structure(residues)


def ideal_helix(n_residues: int = 16, chain: str = "A") -> Structure:
    """Ideal right-handed alpha helix (phi -57, psi -47)."""
    return build_backbone(-57.0, -47.0, n_residues, chain)


def ideal_strand(n_residues: int = 10, chain: str = "A") -> Structure:
    """Ideal extended beta strand (phi -139, psi 135)."""
    return build_backbone(-139.0, 135.0, n_residues, chain)
