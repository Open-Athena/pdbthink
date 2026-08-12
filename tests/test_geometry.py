"""Geometry unit tests with hand-checkable values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pdbthink.chem import MAX_ASA, component_bonds, three_to_one, vdw_radius
from pdbthink.geometry.contacts import find_disulfides, metal_coordination
from pdbthink.geometry.core import (
    circular_difference,
    dihedral,
    distance,
    kabsch,
    normalise_angle,
)
from pdbthink.geometry.dssp import assign_dssp, fold_class
from pdbthink.geometry.rotamer import boundary_distance, chi1_bin, compute_chi1
from pdbthink.geometry.sasa import compute_sasa, sphere_points
from pdbthink.geometry.seqalign import align
from pdbthink.geometry.topology import build_topology
from tests.conftest import ideal_helix, make_residue, make_structure


class TestElementaryGeometry:
    def test_distance_is_euclidean(self):
        assert distance([0, 0, 0], [3, 4, 0]) == pytest.approx(5.0)

    def test_eclipsed_dihedral_is_zero(self):
        # p1 and p4 on the same side of the p2-p3 axis: cis / eclipsed.
        assert dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_anti_dihedral_is_180(self):
        assert abs(dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [-1, 0, 0])) == pytest.approx(180.0)

    def test_right_angle_dihedral(self):
        value = dihedral([1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1])
        assert abs(value) == pytest.approx(90.0)

    @pytest.mark.parametrize(
        "value,expected", [(0, 0), (180, -180), (181, -179), (-180, -180), (359, -1), (-361, -1)]
    )
    def test_angle_normalisation(self, value, expected):
        assert normalise_angle(value) == pytest.approx(expected)

    def test_circular_difference_wraps(self):
        assert circular_difference(170, -170) == pytest.approx(20.0)
        assert circular_difference(-120, 120) == pytest.approx(120.0)


class TestKabsch:
    def test_recovers_a_known_rotation(self):
        rng = np.random.default_rng(0)
        points = rng.normal(size=(12, 3))
        angle = math.radians(37.0)
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        translation = np.array([3.0, -1.0, 0.5])
        moved = (rotation @ points.T).T + translation

        fit = kabsch(points, moved)
        assert fit.rmsd == pytest.approx(0.0, abs=1e-9)
        assert np.allclose(fit.rotation, rotation, atol=1e-9)
        assert np.linalg.det(fit.rotation) == pytest.approx(1.0)

    def test_never_returns_a_reflection(self):
        rng = np.random.default_rng(1)
        points = rng.normal(size=(8, 3))
        mirrored = points * np.array([1.0, 1.0, -1.0])
        fit = kabsch(points, mirrored)
        assert np.linalg.det(fit.rotation) == pytest.approx(1.0, abs=1e-9)


class TestSasa:
    def test_sphere_points_are_on_the_unit_sphere(self):
        points = sphere_points(960)
        assert points.shape == (960, 3)
        assert np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-6)

    def test_isolated_atom_has_full_area(self, definitions):
        residue = make_residue("A", 1, "ALA", {"CB": (0.0, 0.0, 0.0)})
        result = compute_sasa(make_structure([residue]), definitions)
        expected = 4 * math.pi * (vdw_radius("C") + 1.4) ** 2
        assert result.atom_sasa[0] == pytest.approx(expected, rel=1e-9)

    def test_buried_atom_loses_area(self, definitions):
        centre = make_residue("A", 1, "ALA", {"CB": (0.0, 0.0, 0.0)})
        shell = [
            make_residue("A", i + 2, "ALA", {"CB": tuple(3.0 * np.array(v))})
            for i, v in enumerate(
                [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
            )
        ]
        result = compute_sasa(make_structure([centre, *shell]), definitions)
        free = 4 * math.pi * (vdw_radius("C") + 1.4) ** 2
        assert result.atom_sasa[0] < 0.25 * free

    def test_max_asa_table_covers_every_standard_residue(self):
        assert len(MAX_ASA) == 20
        assert MAX_ASA["TRP"] > MAX_ASA["GLY"]


class TestDssp:
    def test_ideal_helix_is_assigned_helix(self, definitions):
        structure = ideal_helix(18)
        result = assign_dssp(structure, definitions)
        interior = [result.three_state[i] for i in range(5, 13)]
        assert interior == ["helix"] * len(interior)

    def test_ideal_helix_classifies_as_alpha(self, definitions):
        structure = ideal_helix(18)
        result = assign_dssp(structure, definitions)
        fractions = result.chain_fractions(structure, "A")
        assert fractions["helix_fraction"] > 0.5
        assert fold_class(fractions, definitions) == "predominantly alpha helical"

    def test_crambin_matches_its_published_fold(self, crambin, definitions):
        result = assign_dssp(crambin.structure, definitions)
        fractions = result.chain_fractions(crambin.structure, "A")
        # Crambin is a small mixed protein with two short helices and a beta hairpin.
        assert fractions["helix_fraction"] > 0.30
        assert fractions["beta_fraction"] > 0.05
        assert result.three_state[crambin.structure.find_index("A:I7")] == "helix"


class TestTopologyAndDisulfides:
    def test_crambin_has_its_three_known_disulfides(self, crambin, definitions):
        found = find_disulfides(crambin.structure, definitions)
        pairs = {
            tuple(
                sorted(
                    (
                        crambin.structure.residues[s.i].seq_id,
                        crambin.structure.residues[s.j].seq_id,
                    )
                )
            )
            for s in found
        }
        assert pairs == {(3, 40), (4, 32), (16, 26)}
        assert all(s.distance <= 2.3 for s in found)

    def test_backbone_bonds_are_one_bond_apart(self, crambin, definitions):
        topology = build_topology(crambin.structure, [])
        index = crambin.structure.index
        n_index = next(
            i
            for i in range(len(index))
            if index.names[i] == "N" and int(index.residue_of[i]) == 1
        )
        ca_index = n_index + 1
        assert topology.separation(n_index, ca_index) == 1

    def test_component_dictionary_covers_the_standard_residues(self):
        for name in ("ALA", "ARG", "TRP", "SEP", "MSE"):
            assert component_bonds(name), name
        assert three_to_one("SEP") == "S"
        assert three_to_one("MSE") == "M"


class TestMetalCoordination:
    def test_carbonic_anhydrase_zinc_has_three_histidines(self, zinc_site, definitions):
        structure = zinc_site.structure
        metal = next(i for i, r in enumerate(structure.residues) if r.name == "ZN")
        result = metal_coordination(structure, metal, definitions)
        residues = [structure.residues[i] for i in result.residues]
        assert sum(1 for r in residues if r.one_letter == "H") >= 3
        assert all(d <= result.cutoff for _, _, d in result.donors)
        assert all(name[0] in "NOS" for _, name, _ in result.donors)


class TestRotamer:
    def test_chi1_bins(self):
        assert chi1_bin(60.0) == "g+"
        assert chi1_bin(-60.0) == "g-"
        assert chi1_bin(179.0) == "t"
        assert chi1_bin(-179.0) == "t"

    def test_boundary_distance_ignores_the_wrap_point(self):
        assert boundary_distance(180.0) == pytest.approx(60.0)
        assert boundary_distance(5.0) == pytest.approx(5.0)

    def test_chi1_is_computed_for_a_real_residue(self, crambin, definitions):
        index = crambin.structure.find_index("A:C3")
        chi = compute_chi1(crambin.structure, index, definitions)
        assert chi is not None
        assert chi.atom4 == "SG"
        assert -180.0 <= chi.angle < 180.0


class TestSequenceAlignment:
    def test_identical_sequences_align_one_to_one(self):
        result = align("ACDEFGHIK", "ACDEFGHIK")
        assert result.identity == pytest.approx(1.0)
        assert result.matched == [(i, i) for i in range(9)]

    def test_gap_is_placed_where_the_deletion_is(self):
        result = align("ACDEFGHIK", "ACDEGHIK")
        assert result.identity == pytest.approx(1.0)
        assert (4, None) in result.pairs      # F is deleted

    def test_low_identity_is_reported(self):
        result = align("ACDEFGHIK", "KIHGFEDCA")
        assert result.identity < 0.5
