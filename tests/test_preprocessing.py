"""Preprocessing tests: A.2 sanitisation, A.3 altlocs, transforms and cropping."""

from __future__ import annotations

import numpy as np
import pytest

from pdbthink.geometry.contacts import min_heavy_distance
from pdbthink.preprocessing import build_transform, display, random_rotation
from pdbthink.preprocessing.crop import crop_around, crop_like
from pdbthink.preprocessing.loader import StructureRejected, load_processed
from pdbthink.preprocessing.transform import TransformError
from pdbthink.representations import render_minimal_pdb, render_table
from tests.conftest import record_for_path

ALTLOC_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  11.400  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      13.200  11.600  10.000  1.00 20.00           O
ATOM      5  CB AALA A   1      12.000   9.200  11.200  0.35 25.00           C
ATOM      6  CB BALA A   1      12.000   9.000  10.900  0.65 25.00           C
ATOM      7  H   ALA A   1      10.200  10.600  10.700  1.00  0.00           H
ATOM      8  N   SER A   2      11.400  12.400  10.000  1.00 22.00           N
ATOM      9  CA  SER A   2      12.000  13.700  10.000  1.00 22.00           C
ATOM     10  C   SER A   2      13.400  13.700  10.500  1.00 22.00           C
ATOM     11  O   SER A   2      14.000  14.800  10.500  1.00 22.00           O
ATOM     12  CB  SER A   2      11.200  14.700  10.800  1.00 22.00           C
ATOM     13  OG  SER A   2      11.700  16.000  10.600  0.00 22.00           O
HETATM   14  O   HOH A 101      20.000  20.000  20.000  1.00 30.00           O
HETATM   15  S   SO4 A 201      25.000  25.000  25.000  1.00 30.00           S
HETATM   16 ZN    ZN A 301      12.500  15.500  11.500  1.00 15.00          ZN
END
"""

ICODE_PDB = """\
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C
ATOM      3  N   GLY A   1A     11.400  12.400  10.000  1.00 22.00           N
ATOM      4  CA  GLY A   1A     12.000  13.700  10.000  1.00 22.00           C
END
"""


@pytest.fixture
def altloc_structure(tmp_path, definitions):
    from pdbthink.config import ProteinSpec

    path = tmp_path / "altloc.pdb"
    path.write_text(ALTLOC_PDB)
    return load_processed(
        record_for_path(path, "TEST"),
        ProteinSpec(id="test", source_type="pdb", entry="TEST"),
        definitions,
    )


class TestSanitisation:
    def test_hydrogens_and_waters_are_removed(self, altloc_structure):
        names = [a.name for r in altloc_structure.structure.residues for a in r.atoms]
        assert "H" not in names
        assert all(r.name != "HOH" for r in altloc_structure.structure.residues)

    def test_crystallisation_additives_are_dropped(self, altloc_structure):
        assert all(r.name != "SO4" for r in altloc_structure.structure.residues)
        assert altloc_structure.stats["dropped_components"].get("SO4") == 1

    def test_metals_keep_their_element_code(self, altloc_structure):
        zinc = [r for r in altloc_structure.structure.residues if r.name == "ZN"]
        assert len(zinc) == 1
        assert zinc[0].label == "A:ZN301"

    def test_occupancy_and_bfactor_are_normalised(self, altloc_structure):
        for residue in altloc_structure.structure.residues:
            for atom in residue.atoms:
                assert atom.occupancy == 1.0
                assert atom.bfactor == 0.0

    def test_atom_serials_are_renumbered_from_one(self, altloc_structure):
        serials = [a.serial for r in altloc_structure.structure.residues for a in r.atoms]
        assert serials == list(range(1, len(serials) + 1))

    def test_insertion_codes_are_rejected(self, tmp_path, definitions):
        from pdbthink.config import ProteinSpec

        path = tmp_path / "icode.pdb"
        path.write_text(ICODE_PDB)
        with pytest.raises(StructureRejected) as excinfo:
            load_processed(
                record_for_path(path, "TEST"),
                ProteinSpec(id="t", source_type="pdb", entry="TEST"),
                definitions,
            )
        assert excinfo.value.reason == "insertion_code_present"


class TestAltloc:
    def test_highest_occupancy_conformer_wins(self, altloc_structure):
        alanine = altloc_structure.structure.find("A:A1")
        cb = alanine.atom("CB")
        # Conformer B has occupancy 0.65 and sits at y = 9.0.
        assert cb.pos[1] == pytest.approx(9.0)

    def test_only_one_conformer_survives(self, altloc_structure):
        alanine = altloc_structure.structure.find("A:A1")
        assert [a.name for a in alanine.atoms].count("CB") == 1

    def test_zero_occupancy_atoms_are_excluded(self, altloc_structure):
        serine = altloc_structure.structure.find("A:S2")
        assert serine.atom("OG") is None


class TestTransforms:
    def test_rotation_is_proper_and_reproducible(self, definitions):
        first = random_rotation(1234, definitions)
        second = random_rotation(1234, definitions)
        assert np.allclose(first, second)
        assert np.linalg.det(first) == pytest.approx(1.0)
        assert np.allclose(first @ first.T, np.eye(3), atol=1e-12)

    def test_different_seeds_give_different_rotations(self, definitions):
        assert not np.allclose(random_rotation(1, definitions), random_rotation(2, definitions))

    def test_distances_survive_the_transform(self, crambin, definitions):
        structure = crambin.structure
        transform = build_transform([structure], 99, definitions)
        moved = display(structure, transform, definitions).structure
        for a, b in (("A:T1", "A:N46"), ("A:I7", "A:V8"), ("A:P19", "A:T30")):
            before, _, _ = min_heavy_distance(structure.find(a), structure.find(b))
            after, _, _ = min_heavy_distance(moved.find(a), moved.find(b))
            assert after == pytest.approx(before, abs=2e-3)

    def test_coordinates_are_rounded_to_three_decimals(self, crambin, definitions):
        transform = build_transform([crambin.structure], 7, definitions)
        moved = display(crambin.structure, transform, definitions).structure
        for residue in moved.residues:
            for atom in residue.atoms:
                for value in atom.pos:
                    assert value == pytest.approx(round(float(value), 3), abs=1e-12)

    def test_overflowing_coordinates_are_rejected(self, definitions):
        from pdbthink.preprocessing.transform import RigidTransform
        from tests.conftest import make_residue, make_structure

        structure = make_structure([make_residue("A", 1, "ALA", {"CA": (0.0, 0.0, 0.0)})])
        transform = RigidTransform(np.eye(3), np.array([1e6, 0.0, 0.0]), 0, "identity")
        with pytest.raises(TransformError):
            display(structure, transform, definitions)


class TestCrop:
    def test_crop_keeps_the_requested_residues(self, crambin):
        structure = crambin.structure
        centre = structure.find_index("A:C16")
        cropped, info = crop_around(structure, [centre], 8.0, required_labels=["A:C16", "A:C26"])
        labels = {r.label for r in cropped.residues}
        assert "A:C16" in labels and "A:C26" in labels
        assert len(cropped.residues) < len(structure.residues)
        assert info.radius == 8.0

    def test_paired_crop_retains_the_same_labels(self, crambin):
        structure = crambin.structure
        centre = structure.find_index("A:C16")
        cropped, info = crop_around(structure, [centre], 8.0)
        mirrored = crop_like(structure, info)
        assert {r.label for r in mirrored.residues} == {r.label for r in cropped.residues}

    def test_crop_records_boundary_residues(self, crambin):
        structure = crambin.structure
        cropped, info = crop_around(structure, [structure.find_index("A:C16")], 8.0)
        assert info.boundary_labels


class TestRepresentationEquivalence:
    def test_pdb_columns_are_fixed_width(self, crambin):
        text = render_minimal_pdb(crambin.structure)
        line = next(ln for ln in text.splitlines() if ln.startswith("ATOM"))
        assert line[0:6].strip() == "ATOM"
        assert line[12:16].strip() == "N"
        assert line[17:20].strip() == "THR"
        assert line[21] == "A"
        assert line[22:26].strip() == "1"
        assert float(line[30:38]) == pytest.approx(crambin.structure.residues[0].atoms[0].pos[0])
        assert line[54:60].strip() == "1.00"
        assert line[60:66].strip() == "0.00"
        assert text.rstrip().endswith("END")

    def test_table_and_pdb_hold_the_same_atoms(self, crambin):
        from pdbthink.representations.table import parse_table

        rows = parse_table(render_table(crambin.structure))
        pdb_atoms = [
            (r.chain, r.seq_id, r.name, a.name)
            for r in crambin.structure.residues
            for a in r.atoms
        ]
        table_atoms = [
            (row["chain"], int(row["resnum"]), row["resname"], row["atom"]) for row in rows
        ]
        assert table_atoms == pdb_atoms

    def test_table_and_pdb_hold_the_same_coordinates(self, crambin):
        from pdbthink.representations.table import parse_table

        rows = parse_table(render_table(crambin.structure))
        text = render_minimal_pdb(crambin.structure)
        pdb_lines = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
        assert len(pdb_lines) == len(rows)
        for line, row in zip(pdb_lines, rows):
            assert float(line[30:38]) == pytest.approx(float(row["x"]))
            assert float(line[38:46]) == pytest.approx(float(row["y"]))
            assert float(line[46:54]) == pytest.approx(float(row["z"]))

    def test_no_annotation_records_are_emitted(self, crambin):
        text = render_minimal_pdb(crambin.structure)
        for banned in ("HELIX", "SHEET", "SSBOND", "LINK", "ANISOU", "HEADER", "REMARK"):
            assert banned not in text
