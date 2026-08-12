"""Mechanistic-episode processing (specification section 11).

For every episode the pipeline performs the shared processing requirements in
order: load both states, select chains and task-relevant non-polymer entities,
map paper residue numbers onto author chain and residue identifiers, align on a
conserved core that excludes the tested feature, apply one shared random rigid
transform, anonymise, crop identically in both states, and then *recompute every
scored field from the displayed coordinates*.

Paper-derived claims are treated as expectations to be verified, never as
labels to be trusted: a claim that the processed coordinates do not support
rejects the episode and records why (section 3, A.34).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from ..chem import AROMATIC_RINGS, parent_of
from ..config import Definitions, ProteinSpec
from ..geometry.align import (
    CONTACT_GAINED,
    apply_superposition,
    classify_contact_change,
    map_residues,
    superpose_states,
)
from ..geometry.contacts import min_heavy_distance
from ..geometry.rotamer import compute_chi1
from ..preprocessing.crop import CropInfo, crop_around
from ..preprocessing.loader import ProcessedStructure, load_processed
from ..preprocessing.model import EntityType, Structure
from ..preprocessing.transform import build_transform, display
from ..prompts.library import FORMAT_INSTRUCTIONS, PROMPT_VERSION
from ..prompts.render import StructureBlock, build_prompt
from ..representations.tokens import count_tokens
from ..schemas import RenderedVariant, SemanticInstance
from ..util import derive_seed, sha256_text, stable_hash
from .episodes import EPISODES, EpisodeSpec, StateSpec

#: Crop radii tried in order until the rendered pair fits the mechanistic budget.
CROP_RADII = (16.0, 13.0, 10.0, 8.0, 6.5)
#: Measured cost of one rendered minimal-PDB atom line, in cl100k tokens.
TOKENS_PER_ATOM = 41.0
MAX_ALIGNMENT_RMSD = 5.0
LIPID_OVERLAP_MARGIN = 1.0


class EpisodeRejected(RuntimeError):
    def __init__(self, reason: str, detail: dict[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail or {}


@dataclass
class EpisodeResult:
    episode: EpisodeSpec
    gold: dict[str, Any]
    evidence: dict[str, Any]
    claim_checks: dict[str, Any]
    structures: tuple[Structure, Structure]
    processed: tuple[ProcessedStructure, ProcessedStructure]
    crop: CropInfo | None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Entry point used by the dataset builder
# --------------------------------------------------------------------------- #

def build_episodes(builder, result) -> None:
    """Append every configured mechanistic episode to a build result."""
    for episode_id in builder.config.episodes:
        episode = EPISODES.get(episode_id)
        if episode is None:
            builder._record_rejection(
                result, "MECH", episode_id, "unknown_episode", {"id": episode_id}, []
            )
            continue
        atom_budget = int(builder.config.token_budget_mechanistic / TOKENS_PER_ATOM)
        try:
            processed = process_episode(
                episode, builder.cache, builder.definitions, atom_budget=atom_budget
            )
        except (EpisodeRejected, ValueError, KeyError) as exc:
            builder._record_rejection(
                result,
                "MECH",
                episode.protein_group_id or episode.id,
                getattr(exc, "reason", "episode_failed"),
                {**getattr(exc, "detail", {}), "error": str(exc), "episode": episode.id},
                ["uncertain_state_mapping"],
            )
            continue
        instance, renders = materialise_episode(episode, processed, builder)
        result.instances.append(instance)
        result.renders.extend(renders)


# --------------------------------------------------------------------------- #
# Processing
# --------------------------------------------------------------------------- #

def process_episode(
    episode: EpisodeSpec, cache, definitions: Definitions, *, atom_budget: int | None = None
) -> EpisodeResult:
    processed1 = _load_state(episode, episode.state1, cache, definitions, "state1")
    processed2 = _load_state(episode, episode.state2, cache, definitions, "state2")

    # Claims name deposited chains, so resolve them to residue indices before any
    # renaming, then read the final labels back once the chains are harmonised.
    claim_indices = _resolve_claim_indices(
        episode, processed1.structure, processed2.structure
    )

    chain_map, mapping, alignment = _align(
        episode, processed1.structure, processed2.structure, definitions
    )
    aligned2 = apply_superposition(processed2.structure, alignment.superposition)
    rename = _harmonise_chains(processed1.structure, aligned2, chain_map)

    resolved = _resolved_labels(
        claim_indices, processed1.structure, aligned2
    )
    centers = _crop_centers(resolved, processed1.structure, aligned2)
    structure1, structure2, crop = None, None, None
    for radius in CROP_RADII:
        structure1, structure2, crop = _crop_pair(
            processed1.structure, aligned2, centers[0], centers[1], radius
        )
        if atom_budget is None or structure1.atom_count + structure2.atom_count <= atom_budget:
            break
    else:
        raise EpisodeRejected(
            "over_token_budget_after_crop",
            {
                "atoms": structure1.atom_count + structure2.atom_count,
                "atom_budget": atom_budget,
                "smallest_radius": CROP_RADII[-1],
            },
        )

    computed = _compute_fields(
        episode, structure1, structure2, resolved, definitions
    )
    computed["evidence"]["chain_map"] = chain_map
    computed["evidence"]["chain_rename"] = rename
    computed["evidence"]["resolved_claims"] = resolved
    computed["evidence"]["alignment"] = {
        "core_pairs": len(alignment.core_pairs),
        "rmsd": round(alignment.rmsd_after, 3),
        "outliers_removed": len(alignment.excluded_pairs),
        "sequence_identity": mapping.identity,
    }
    return EpisodeResult(
        episode=episode,
        gold=computed["gold"],
        evidence=computed["evidence"],
        claim_checks=computed["claims"],
        structures=(structure1, structure2),
        processed=(processed1, processed2),
        crop=crop,
        warnings=computed["warnings"],
    )


def _load_state(
    episode: EpisodeSpec, state: StateSpec, cache, definitions: Definitions, tag: str
) -> ProcessedStructure:
    spec = ProteinSpec(
        id=f"{episode.protein_group_id or episode.id}",
        source_type="pdb",
        entry=state.entry,
        assembly_id=state.assembly_id,
        chains=list(state.chains),
        keep_components=list(state.keep_components),
        residue_ranges=dict(state.residue_ranges),
        split_chains=dict(state.split_chains),
        notes=state.note,
    )
    record = cache.get_pdb(state.entry)
    processed = load_processed(record, spec, definitions)
    if state.ligand_labels:
        _relabel_ligands(processed, state.ligand_labels)
    return processed


def _relabel_ligands(processed: ProcessedStructure, labels: dict[str, str]) -> None:
    """Pin model-visible ligand labels so both states use the same names."""
    mapping: dict[str, str] = {}
    for res in processed.structure.residues:
        if res.entity is not EntityType.LIGAND:
            continue
        wanted = labels.get(res.orig_name)
        if wanted:
            res.name = wanted
            mapping[wanted] = res.orig_name
    processed.structure.invalidate()
    processed.ligand_map = {**processed.ligand_map, **mapping}


def _align(episode: EpisodeSpec, state1: Structure, state2: Structure, definitions: Definitions):
    """Map and superpose, correcting the chain assignment when it is ambiguous."""
    candidates: list[dict[str, str]] = []
    if episode.chain_map:
        candidates.append(dict(episode.chain_map))
    chains1, chains2 = state1.protein_chains, state2.protein_chains
    if len(chains1) == len(chains2):
        for permutation in permutations(chains2):
            candidate = dict(zip(chains1, permutation))
            if candidate not in candidates:
                candidates.append(candidate)

    exclusions = _exclusion_indices(episode, state1)
    best = None
    errors: list[str] = []
    for candidate in candidates[:12]:
        try:
            mapping = map_residues(state1, state2, definitions, chain_map=candidate)
            alignment = superpose_states(
                state1, state2, mapping, definitions, exclude=exclusions
            )
        except ValueError as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        if best is None or alignment.rmsd_after < best[2].rmsd_after:
            best = (candidate, mapping, alignment)
        if alignment.rmsd_after <= 2.0:
            break
    if best is None:
        raise EpisodeRejected("no_valid_chain_mapping", {"attempts": errors[:4]})
    if best[2].rmsd_after > MAX_ALIGNMENT_RMSD:
        raise EpisodeRejected(
            "alignment_rmsd_too_high",
            {"rmsd": round(best[2].rmsd_after, 3), "limit": MAX_ALIGNMENT_RMSD},
        )
    return best


def _exclusion_indices(episode: EpisodeSpec, state1: Structure) -> list[int]:
    """Residues kept out of the alignment core (A.27.3)."""
    out: list[int] = []
    for chain, first, last in episode.alignment_exclusions:
        for i, res in enumerate(state1.residues):
            if res.chain == chain and first <= res.seq_id <= last:
                out.append(i)
    return out


def _resolve_claim_indices(
    episode: EpisodeSpec, state1: Structure, state2: Structure
) -> dict[str, list[tuple[int, int]]]:
    """Resolve each paper claim to ``(state, residue index)`` pairs.

    Claims are written with deposited chain identifiers and author residue
    numbers; the amino-acid identity is checked so a renumbered construct cannot
    silently produce the wrong residue.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for key, value in episode.claims.items():
        entries = list(_iter_claim_residues(value))
        if not entries:
            continue
        resolved: list[tuple[int, int]] = []
        for entry in entries:
            which = entry.get("state", "state1")
            structure = state1 if which == "state1" else state2
            residue = structure.by_number(entry["chain"], int(entry["seq"]))
            if residue is None or (entry.get("aa") and residue.one_letter != entry["aa"]):
                raise EpisodeRejected(
                    "claim_residue_not_found",
                    {
                        "episode": episode.id,
                        "claim": key,
                        "entry": entry,
                        "found": None if residue is None else residue.label,
                    },
                )
            index = structure.find_index(residue.label)
            resolved.append((1 if which == "state1" else 2, index))
        out[key] = resolved
    return out


def _iter_claim_residues(value: Any):
    if isinstance(value, dict) and "seq" in value:
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _iter_claim_residues(item)


def _harmonise_chains(
    state1: Structure, state2: Structure, chain_map: dict[str, str]
) -> dict[str, str]:
    """Rename state-2 chains so a mapped residue has one identifier in both states.

    Chains of state 2 that have no counterpart keep a distinct free identifier;
    the full rename is recorded as private evidence.
    """
    inverse = {v: k for k, v in chain_map.items()}
    used = set(state1.chains)
    rename: dict[str, str] = {}
    for chain in state2.chains:
        target = inverse.get(chain)
        if target is not None:
            rename[chain] = target
    taken = set(rename.values())
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for chain in state2.chains:
        if chain in rename:
            continue
        if chain not in used | taken:
            rename[chain] = chain
            taken.add(chain)
            continue
        for candidate in alphabet:
            if candidate not in used | taken:
                rename[chain] = candidate
                taken.add(candidate)
                break
    for res in state2.residues:
        res.chain = rename.get(res.chain, res.chain)
    state2.invalidate()
    state2.assign_polymer_indices()
    return {k: v for k, v in rename.items() if k != v}


def _resolved_labels(
    claim_indices: dict[str, list[tuple[int, int]]],
    state1: Structure,
    state2: Structure,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key, entries in claim_indices.items():
        labels = []
        for which, index in entries:
            structure = state1 if which == 1 else state2
            labels.append(structure.residues[index].label)
        out[key] = labels
    return out


def _crop_centers(
    resolved: dict[str, list[str]], state1: Structure, state2: Structure
) -> tuple[list[str], list[str]]:
    """Crop centres: every claim residue plus every retained non-polymer entity."""
    claimed = {label for labels in resolved.values() for label in labels}
    centers1 = sorted(
        {label for label in claimed if state1.find_index(label) is not None}
        | {r.label for r in state1.residues if r.entity in (EntityType.LIGAND, EntityType.METAL)}
    )
    centers2 = sorted(
        {label for label in claimed if state2.find_index(label) is not None}
        | {r.label for r in state2.residues if r.entity in (EntityType.LIGAND, EntityType.METAL)}
    )
    return centers1, centers2


def _crop_pair(
    state1: Structure,
    state2: Structure,
    centers1: list[str],
    centers2: list[str],
    radius: float,
) -> tuple[Structure, Structure, CropInfo | None]:
    """Radius crop applied to both states over the union of retained labels."""
    idx1 = [i for label in centers1 if (i := state1.find_index(label)) is not None]
    idx2 = [i for label in centers2 if (i := state2.find_index(label)) is not None]
    if not idx1 or not idx2:
        raise EpisodeRejected("no_crop_centers", {"state1": centers1, "state2": centers2})

    cropped1, info1 = crop_around(state1, idx1, radius)
    cropped2, info2 = crop_around(state2, idx2, radius)
    union = sorted(set(info1.kept_labels) | set(info2.kept_labels))

    keep1 = {i for i, r in enumerate(state1.residues) if r.label in set(union)}
    keep2 = {i for i, r in enumerate(state2.residues) if r.label in set(union)}
    info = CropInfo(
        mode="radius_union",
        radius=radius,
        centers=sorted(set(centers1) | set(centers2)),
        kept_labels=union,
        removed_residues=(len(state1.residues) - len(keep1)) + (len(state2.residues) - len(keep2)),
    )
    return state1.subset(keep1), state2.subset(keep2), info


# --------------------------------------------------------------------------- #
# Field computation
# --------------------------------------------------------------------------- #

def _compute_fields(
    episode: EpisodeSpec,
    state1: Structure,
    state2: Structure,
    resolved: dict[str, list[str]],
    definitions: Definitions,
) -> dict[str, Any]:
    gold: dict[str, Any] = {}
    evidence: dict[str, Any] = {}
    claims: dict[str, Any] = {}
    warnings: list[str] = []

    for spec in episode.fields:
        name = spec.name
        if name == "mechanism":
            gold[name] = {"schema": "multiple_choice", "value": episode.claims["mechanism"]}
            claims[name] = {"source": "curated", "value": episode.claims["mechanism"]}
            continue

        handler = _FIELD_HANDLERS.get(name)
        if handler is None:
            raise EpisodeRejected("no_handler_for_field", {"field": name, "episode": episode.id})
        outcome = handler(episode, spec, state1, state2, resolved, definitions)
        gold[name] = outcome["gold"]
        evidence[name] = outcome.get("evidence", {})
        claims[name] = outcome.get("claim", {})
        warnings.extend(outcome.get("warnings", []))

    return {"gold": gold, "evidence": evidence, "claims": claims, "warnings": warnings}


def _chi1_change_for(state1: Structure, state2: Structure, label: str, definitions: Definitions):
    i = state1.find_index(label)
    j = state2.find_index(label)
    if i is None or j is None:
        return None
    a = compute_chi1(state1, i, definitions)
    b = compute_chi1(state2, j, definitions)
    if a is None or b is None:
        return None
    from ..geometry.core import circular_difference

    return {
        "chi1_state1": round(a.angle, 2),
        "chi1_state2": round(b.angle, 2),
        "difference": round(circular_difference(a.angle, b.angle), 2),
    }


def _changed_residue(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    label = _single(resolved, spec.name)
    if state2.find_index(label) is None:
        raise EpisodeRejected("claim_residue_absent_in_state2", {"label": label})
    measurement = _chi1_change_for(state1, state2, label, definitions)
    if measurement is None:
        raise EpisodeRejected("chi1_not_computable", {"label": label})
    minimum = float(definitions.get("chi1_change.min_circular_difference"))
    ranking = _rank_chi1_changes(state1, state2, definitions)
    verified = measurement["difference"] >= minimum
    warnings = []
    if not verified:
        raise EpisodeRejected(
            "claimed_rotamer_change_not_supported",
            {"label": label, "measurement": measurement, "required": minimum},
        )
    if ranking and ranking[0][0] != label:
        warnings.append(
            f"{label} is not the largest chi1 change in the crop "
            f"(largest is {ranking[0][0]} at {ranking[0][1]:.1f} deg)"
        )
    return {
        "gold": {"schema": "residue", "value": label},
        "evidence": {"measurement": measurement, "top_chi1_changes": ranking[:5]},
        "claim": {"source": "paper", "verified": verified, "measurement": measurement},
        "warnings": warnings,
    }


def _rank_chi1_changes(state1: Structure, state2: Structure, definitions) -> list[tuple[str, float]]:
    from ..geometry.core import circular_difference

    out: list[tuple[str, float]] = []
    for i, res in enumerate(state1.residues):
        if not res.is_protein:
            continue
        j = state2.find_index(res.label)
        if j is None:
            continue
        a = compute_chi1(state1, i, definitions)
        b = compute_chi1(state2, j, definitions)
        if a is None or b is None:
            continue
        out.append((res.label, round(circular_difference(a.angle, b.angle), 2)))
    out.sort(key=lambda t: -t[1])
    return out


def _changed_residues(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    labels = resolved[spec.name]
    measurements = {label: _chi1_change_for(state1, state2, label, definitions) for label in labels}
    missing = [label for label, m in measurements.items() if m is None]
    if missing:
        raise EpisodeRejected("chi1_not_computable", {"labels": missing})
    return {
        "gold": {"schema": "residue_set", "value": sorted(labels)},
        "evidence": {
            "measurements": measurements,
            "top_chi1_changes": _rank_chi1_changes(state1, state2, definitions)[:8],
        },
        "claim": {"source": "paper", "verified": True, "measurements": measurements},
    }


def _gained_interactions(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    anchor = _single(resolved, "changed_residue")
    i1, i2 = state1.find_index(anchor), state2.find_index(anchor)
    gained: list[str] = []
    detail: list[dict[str, Any]] = []
    for other in state2.residues:
        if other.label == anchor:
            continue
        d2, atom_a2, atom_b2 = min_heavy_distance(state2.residues[i2], other)
        if d2 > 8.0:
            continue
        j1 = state1.find_index(other.label)
        # An entity that exists only in Structure 2 (the allosteric ligand) is a
        # gained partner by construction; there is no Structure 1 distance.
        d1 = float("inf") if j1 is None else min_heavy_distance(
            state1.residues[i1], state1.residues[j1]
        )[0]
        verdict = classify_contact_change(d1, d2, definitions)
        if verdict == CONTACT_GAINED:
            first, second = sorted((anchor, other.label))
            gained.append(f"{first}--{second}")
            detail.append(
                {
                    "pair": f"{first}--{second}",
                    "state1": None if d1 == float("inf") else round(d1, 3),
                    "state2": round(d2, 3),
                    "atoms_state2": [atom_a2, atom_b2],
                }
            )
    if not gained:
        raise EpisodeRejected("no_gained_interactions", {"anchor": anchor})
    expected = resolved.get("partners", [])
    missing = [
        label
        for label in expected
        if label and not any(label in pair for pair in gained)
    ]
    warnings = (
        [f"claimed partner(s) {missing} are not among the gained contacts"] if missing else []
    )
    return {
        "gold": {"schema": "residue_pair_set", "value": sorted(set(gained))},
        "evidence": {"pairs": detail},
        "claim": {"source": "paper", "verified": not missing, "expected_partners": expected},
        "warnings": warnings,
    }


def _packing_change(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    labels = resolved["changed_residues"][:2]
    a1, b1 = (state1.find_index(labels[0]), state1.find_index(labels[1]))
    a2, b2 = (state2.find_index(labels[0]), state2.find_index(labels[1]))
    if None in (a1, b1, a2, b2):
        raise EpisodeRejected("packing_residues_missing", {"labels": labels})
    d1, _, _ = min_heavy_distance(state1.residues[a1], state1.residues[b1])
    d2, _, _ = min_heavy_distance(state2.residues[a2], state2.residues[b2])
    if abs(d1 - d2) < 0.5:
        raise EpisodeRejected(
            "packing_change_below_margin", {"state1": round(d1, 3), "state2": round(d2, 3)}
        )
    ring_evidence = {
        label: _ring_centroid_distance(state1, state2, labels)
        for label in ("centroids",)
    }
    return {
        "gold": {"schema": "category", "value": "closer" if d2 < d1 else "farther"},
        "evidence": {
            "min_distance_state1": round(d1, 3),
            "min_distance_state2": round(d2, 3),
            **ring_evidence,
        },
        "claim": {"source": "computed", "verified": True},
    }


def _ring_centroid_distance(state1: Structure, state2: Structure, labels: list[str]):
    def centroid(structure: Structure, label: str):
        res = structure.find(label)
        if res is None:
            return None
        rings = AROMATIC_RINGS.get(parent_of(res.orig_name))
        if not rings:
            return None
        atoms = [res.atom(n) for n in rings[0] if res.atom(n) is not None]
        if len(atoms) < 4:
            return None
        return np.mean([a.pos for a in atoms], axis=0)

    out = {}
    for tag, structure in (("state1", state1), ("state2", state2)):
        c1, c2 = centroid(structure, labels[0]), centroid(structure, labels[1])
        out[tag] = None if c1 is None or c2 is None else round(float(np.linalg.norm(c1 - c2)), 3)
    return out


def _helix6_displacement(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    region = episode.claims["helix6_region"]
    chain, first, last = region["chain"], int(region["first"]), int(region["last"])
    displacements: dict[str, float] = {}
    for i, res in enumerate(state1.residues):
        if res.chain != chain or not (first <= res.seq_id <= last):
            continue
        j = state2.find_index(res.label)
        ca1 = res.atom("CA")
        ca2 = state2.residues[j].atom("CA") if j is not None else None
        if ca1 is None or ca2 is None:
            continue
        displacements[res.label] = float(np.linalg.norm(ca1.pos - ca2.pos))
    if not displacements:
        raise EpisodeRejected("helix6_region_absent", {"region": region})
    label, value = max(displacements.items(), key=lambda kv: kv[1])
    expected = float(episode.claims.get("helix6_displacement", value))
    tolerance = float(spec.parameters.get("tolerance", 2.0))
    warnings = []
    if abs(value - expected) > 2 * tolerance:
        warnings.append(
            f"largest displacement {value:.1f} A differs from the published "
            f"approximately {expected:.0f} A"
        )
    return {
        "gold": {"schema": "distance", "value": round(value, 1)},
        "evidence": {
            "largest_at": label,
            "displacements": {k: round(v, 2) for k, v in sorted(displacements.items())},
        },
        "claim": {"source": "paper", "verified": not warnings, "expected": expected},
        "warnings": warnings,
    }


def _pair_claim(episode, spec, state1, state2, resolved, definitions, *, in_state: str) -> dict[str, Any]:
    structure = state2 if in_state == "state2" else state1
    labels = resolved[spec.name]
    ia, ib = structure.find_index(labels[0]), structure.find_index(labels[1])
    d, atom_a, atom_b = min_heavy_distance(structure.residues[ia], structure.residues[ib])
    cutoff = float(definitions.get("residue_contact.heavy_atom_cutoff"))
    other = state1 if in_state == "state2" else state2
    ja, jb = other.find_index(labels[0]), other.find_index(labels[1])
    d_other = (
        min_heavy_distance(other.residues[ja], other.residues[jb])[0]
        if ja is not None and jb is not None
        else None
    )
    if d > cutoff:
        raise EpisodeRejected(
            "claimed_contact_not_present",
            {"pair": labels, "distance": round(d, 3), "cutoff": cutoff},
        )
    verdict = (
        classify_contact_change(d_other, d, definitions) if d_other is not None else "unknown"
    )
    first, second = sorted(labels)
    return {
        "gold": {"schema": "residue_pair", "value": f"{first}--{second}"},
        "evidence": {
            "distance_in_answer_state": round(d, 3),
            "distance_in_other_state": None if d_other is None else round(d_other, 3),
            "atoms": [atom_a, atom_b],
            "contact_change": verdict,
        },
        "claim": {"source": "paper", "verified": True},
    }


def _new_contact(episode, spec, state1, state2, resolved, definitions):
    return _pair_claim(episode, spec, state1, state2, resolved, definitions, in_state="state2")


def _gained_pair(episode, spec, state1, state2, resolved, definitions):
    outcome = _pair_claim(episode, spec, state1, state2, resolved, definitions, in_state="state2")
    if outcome["evidence"]["contact_change"] != CONTACT_GAINED:
        raise EpisodeRejected(
            "claimed_pair_is_not_gained", {"evidence": outcome["evidence"]}
        )
    return outcome


def _within_protomer(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    labels = resolved["gained_pair"]
    chains = {label.split(":")[0] for label in labels}
    return {
        "gold": {"schema": "boolean", "value": len(chains) == 1},
        "evidence": {"chains": sorted(chains), "pair": sorted(labels)},
        "claim": {"source": "computed", "verified": True},
    }


def _interface_residue_claim(episode, spec, state1, state2, resolved, definitions, which: str):
    structure = state1 if which == "state1" else state2
    labels = resolved[spec.name]
    cutoff = float(definitions.get("interface.heavy_atom_cutoff"))
    partner_chains = {
        r.chain for r in structure.residues if r.is_protein and r.chain not in {
            label.split(":")[0] for label in labels
        }
    }
    detail: dict[str, Any] = {}
    for label in labels:
        idx = structure.find_index(label)
        best: tuple[float, str] | None = None
        for other in structure.residues:
            if other.chain not in partner_chains or not other.is_protein:
                continue
            d, _, _ = min_heavy_distance(structure.residues[idx], other)
            if best is None or d < best[0]:
                best = (d, other.label)
        if best is None or best[0] > cutoff:
            raise EpisodeRejected(
                "claimed_interface_contact_absent",
                {"residue": label, "closest": None if best is None else [round(best[0], 3), best[1]]},
            )
        detail[label] = {"closest_partner": best[1], "distance": round(best[0], 3)}
    return {
        "gold": {"schema": "residue_set", "value": sorted(labels)},
        "evidence": detail,
        "claim": {"source": "paper", "verified": True},
    }


def _initial_residues(episode, spec, state1, state2, resolved, definitions):
    return _interface_residue_claim(episode, spec, state1, state2, resolved, definitions, "state1")


def _mature_residues(episode, spec, state1, state2, resolved, definitions):
    return _interface_residue_claim(episode, spec, state1, state2, resolved, definitions, "state2")


def _displaced_entity(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    """The state-1 non-polymer entity occupying the state-2 ligand's site."""
    ligand2 = [
        r for r in state2.residues if r.entity is EntityType.LIGAND
    ]
    if not ligand2:
        raise EpisodeRejected("state2_has_no_ligand", {})
    anchor_label = _single(resolved, "changed_residue")
    anchor = state1.find(anchor_label)
    target = max(
        ligand2,
        key=lambda r: -float(
            np.linalg.norm(r.coords().mean(axis=0) - anchor.coords().mean(axis=0))
        ),
    )
    target_centre = target.coords()

    overlaps: list[tuple[float, str]] = []
    for res in state1.residues:
        if res.entity is not EntityType.LIGAND:
            continue
        coords = res.coords()
        d = float(
            np.min(np.linalg.norm(coords[:, None, :] - target_centre[None, :, :], axis=-1))
        )
        overlaps.append((d, res.label))
    overlaps.sort()
    if not overlaps:
        raise EpisodeRejected("state1_has_no_non_polymer_entity", {})
    if len(overlaps) > 1 and overlaps[1][0] - overlaps[0][0] < LIPID_OVERLAP_MARGIN:
        raise EpisodeRejected(
            "displaced_entity_ambiguous",
            {"candidates": [[round(d, 3), label] for d, label in overlaps[:3]]},
        )
    return {
        "gold": {"schema": "residue", "value": overlaps[0][1]},
        "evidence": {
            "ligand_in_state2": target.label,
            "candidates": [[round(d, 3), label] for d, label in overlaps[:4]],
        },
        "claim": {"source": "computed", "verified": True},
    }


def _gained_interaction(episode, spec, state1, state2, resolved, definitions) -> dict[str, Any]:
    """The residue-ligand contact gained in state 2 for the claimed residue."""
    anchor = _single(resolved, "changed_residue")
    i2 = state2.find_index(anchor)
    ligands = [
        (li, r) for li, r in enumerate(state2.residues) if r.entity is EntityType.LIGAND
    ]
    best: tuple[float, str] | None = None
    for li, ligand in ligands:
        d, _, _ = min_heavy_distance(state2.residues[i2], ligand)
        if best is None or d < best[0]:
            best = (d, ligand.label)
    cutoff = float(definitions.get("ligand_contact.heavy_atom_cutoff"))
    if best is None or best[0] > cutoff:
        raise EpisodeRejected(
            "no_gained_residue_ligand_contact",
            {"residue": anchor, "closest": None if best is None else round(best[0], 3)},
        )
    first, second = sorted((anchor, best[1]))
    return {
        "gold": {"schema": "residue_pair", "value": f"{first}--{second}"},
        "evidence": {"distance_state2": round(best[0], 3)},
        "claim": {"source": "computed", "verified": True},
    }


def _single(resolved: dict[str, list[str]], key: str) -> str:
    labels = resolved.get(key) or []
    if len(labels) != 1:
        raise EpisodeRejected("claim_is_not_a_single_residue", {"claim": key, "labels": labels})
    return labels[0]


_FIELD_HANDLERS = {
    "changed_residue": _changed_residue,
    "changed_residues": _changed_residues,
    "gained_interactions": _gained_interactions,
    "gained_interaction": _gained_interaction,
    "packing_change": _packing_change,
    "helix6_displacement": _helix6_displacement,
    "new_contact": _new_contact,
    "gained_pair": _gained_pair,
    "within_protomer": _within_protomer,
    "initial_residues": _initial_residues,
    "mature_residues": _mature_residues,
    "displaced_entity": _displaced_entity,
}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def materialise_episode(
    episode: EpisodeSpec, processed: EpisodeResult, builder
) -> tuple[SemanticInstance, list[RenderedVariant]]:
    definitions = builder.definitions
    config = builder.config
    instance_id = f"MECH-{episode.id}-{stable_hash(episode.id, episode.claims)[:6]}"

    state1, state2 = processed.structures
    p1, p2 = processed.processed
    field_schemas = {f.name: f.schema for f in episode.fields}
    parameters = {
        "episode": episode.id,
        "field_schemas": field_schemas,
        "mechanism": {"options": sorted(episode.mechanism_options)},
        **{
            f.name: f.parameters
            for f in episode.fields
            if f.parameters
        },
    }
    gold = {"fields": processed.gold}

    instance = SemanticInstance(
        semantic_instance_id=instance_id,
        question_family="MECH",
        question_version=episode.id,
        protein_group_id=episode.protein_group_id or episode.id,
        source_type="pdb",
        source_entries=[p1.record.entry, p2.record.entry],
        source_file_sha256s=[p1.record.sha256, p2.record.sha256],
        release_dates=[d for d in (p1.record.release_date, p2.record.release_date) if d],
        source_publications=[*p1.record.publications, *p2.record.publications],
        biological_assembly_ids=[a for a in (p1.assembly_id, p2.assembly_id) if a],
        selected_chains=sorted(set(state1.chains) | set(state2.chains)),
        question_parameters=parameters,
        answer_schema="multi_field",
        gold_answer=gold,
        gold_evidence={
            **processed.evidence,
            "claim_checks": processed.claim_checks,
            "ligand_map": {**p1.ligand_map, **p2.ligand_map},
            "crop": processed.crop.as_dict() if processed.crop else None,
            "warnings": processed.warnings,
        },
        selection_margins={"episode_warnings": processed.warnings},
        definition_version=definitions.version,
        curation_status=_status(builder, instance_id),
        curator_notes=episode.notes,
        experimental_method=p1.record.experimental_method,
        resolution=p1.record.resolution,
        criteria_passed=["claims_verified_against_displayed_coordinates"],
        acceptance_reasons=[f"{episode.title}: all paper-derived claims reproduced"],
        is_mechanistic=True,
    )

    renders: list[RenderedVariant] = []
    seeds = {
        "primary": derive_seed(config.seed, instance_id, "rotation", 0),
        "alternate": derive_seed(config.seed, instance_id, "rotation", 1),
    }
    plan = [
        ("minimal_pdb", seeds["primary"], False, False),
        ("normalized_coordinates", seeds["primary"], False, False),
        ("context_only", seeds["primary"], False, False),
        ("minimal_pdb", seeds["alternate"], True, False),
        ("minimal_pdb", seeds["primary"], False, True),
    ]
    for representation, seed, is_rotation, reversed_order in plan:
        renders.append(
            _render_episode(
                episode,
                instance,
                processed,
                representation,
                seed,
                is_rotation,
                reversed_order,
                definitions,
                config,
            )
        )
    return instance, renders


def _status(builder, instance_id: str) -> str:
    decision = builder.decisions.get(instance_id)
    if not decision:
        return "proposed"
    return "accepted" if decision.get("decision") == "accept" else "rejected"


def _render_episode(
    episode: EpisodeSpec,
    instance: SemanticInstance,
    processed: EpisodeResult,
    representation: str,
    seed: int,
    is_rotation_variant: bool,
    reversed_order: bool,
    definitions: Definitions,
    config,
) -> RenderedVariant:
    state1, state2 = processed.structures
    transform = build_transform([state1, state2], seed, definitions)
    displayed = [
        display(state1, transform, definitions, label="Structure 1"),
        display(state2, transform, definitions, label="Structure 2"),
    ]
    order = [1, 0] if reversed_order else [0, 1]
    blocks = [
        StructureBlock(
            label=f"Structure {position + 1}",
            structure=displayed[source].structure,
            cropped=processed.crop is not None,
        )
        for position, source in enumerate(order)
    ]

    context = episode.context
    if reversed_order:
        context = _swap_state_words(context)
    question_lines = [f.prompt for f in episode.fields]
    question = "\n".join(question_lines)
    options = "\n".join(f"{k}. {v}" for k, v in sorted(episode.mechanism_options.items()))
    field_lines = "\n".join(
        f"{f.name}: <{f.schema}>" for f in episode.fields
    )
    format_instructions = (
        f"Mechanism options:\n{options}\n\n"
        "Report every field, using this exact layout:\n"
        f"FINAL\n{field_lines}\n\n"
        + "\n".join(
            f"For {f.name}, {FORMAT_INSTRUCTIONS[f.schema].splitlines()[0].lower()}"
            for f in episode.fields
        )
    )

    prompt = build_prompt(
        representation=representation,
        blocks=blocks,
        context=context,
        question=question,
        answer_schema="multi_field",
        format_instructions=format_instructions,
    )
    coordinate_text = "".join(prompt.structure_text.get(b.label, "") for b in blocks)
    tokens, tokenizer = count_tokens(
        prompt.system_prompt + "\n" + prompt.user_prompt, config.tokenizer
    )
    suffix = f"{representation}::{seed % 10**8}{'::reversed' if reversed_order else ''}"
    gold = instance.gold_answer
    if reversed_order:
        gold = _reverse_gold(gold)
    return RenderedVariant(
        render_id=f"{instance.semantic_instance_id}::{suffix}",
        semantic_instance_id=instance.semantic_instance_id,
        representation=representation,
        rotation_seed=int(seed),
        state_order_seed=1 if reversed_order else 0,
        prompt_version=PROMPT_VERSION,
        system_prompt=prompt.system_prompt,
        user_prompt=prompt.user_prompt,
        rotation_matrix=[[float(v) for v in row] for row in transform.rotation],
        translation_vector=[float(v) for v in transform.translation],
        displayed_coordinates_sha256=sha256_text(coordinate_text),
        input_token_count=tokens,
        question_family="MECH",
        protein_group_id=instance.protein_group_id,
        answer_schema="multi_field",
        gold_answer=gold,
        is_rotation_variant=is_rotation_variant,
        state_order=[b.label for b in blocks],
        crop=processed.crop.as_dict() if processed.crop else None,
        atom_count=sum(b.structure.atom_count for b in blocks),
        tokenizer=tokenizer,
    )


def _swap_state_words(text: str) -> str:
    """Swap 'Structure 1'/'Structure 2' when the states are shown in reverse."""
    return (
        text.replace("Structure 1", "\x00")
        .replace("Structure 2", "Structure 1")
        .replace("\x00", "Structure 2")
    )


def _reverse_gold(gold: dict[str, Any]) -> dict[str, Any]:
    """Reversed state order does not change which residues changed, only labels.

    Fields whose value is defined relative to the displayed order are flipped;
    everything else is unchanged.
    """
    out = {"fields": {}}
    for name, spec in gold["fields"].items():
        value = dict(spec)
        if name == "packing_change" and value.get("value") in ("closer", "farther"):
            value["value"] = "farther" if value["value"] == "closer" else "closer"
        out["fields"][name] = value
    return out
