"""Statistics for the benchmark report (specification section 13).

Three sources of variation are kept separate:

1. benchmark composition -> hierarchical bootstrap clustered by protein or paper,
2. model stochasticity   -> repeated completions of identical prompts,
3. representation        -> matched minimal-PDB / normalized / context-only /
   rotation variants of the same semantic instance.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from statistics import mean
from typing import Any

import numpy as np

from ..util import rng_for

SET_SCHEMAS = ("residue_set", "string_set", "residue_pair_set", "two_interaction_sets")


def macro_score(rows: Sequence[dict[str, Any]]) -> float:
    """Mean over question families of the mean score within each family."""
    per_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_family[row["question_family"]].append(row["score"])
    if not per_family:
        return 0.0
    return mean(mean(v) for v in per_family.values())


def micro_score(rows: Sequence[dict[str, Any]]) -> float:
    return mean(r["score"] for r in rows) if rows else 0.0


def instance_scores(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Mean score per semantic instance, averaged over completions and variants.

    Section 13: the primary model score is the mean across completions, never
    pass@k.
    """
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["semantic_instance_id"]].append(row["score"])
    return {k: mean(v) for k, v in grouped.items()}


def group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key))].append(row)
    return dict(sorted(out.items()))


def summarise_group(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "n_instances": len({r["semantic_instance_id"] for r in rows}),
        "mean_score": micro_score(rows),
        "exact": mean(1.0 if r["correct"] else 0.0 for r in rows) if rows else 0.0,
        "format_errors": sum(1 for r in rows if r["format_error"]),
        "refusals": sum(1 for r in rows if r["refusal"]),
        "truncated": sum(1 for r in rows if r["truncated"]),
    }


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def clustered_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    samples: int = 2000,
    cluster_key: str = "cluster",
    seed: Any = "bootstrap",
    statistic=macro_score,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile confidence interval, resampling whole protein clusters.

    Clustering by protein (or by source paper, when several structures share
    one) is what keeps the interval honest: instances from the same structure
    are not independent draws from the benchmark's composition.
    """
    if not rows:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_clusters": 0, "samples": 0}
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cluster[str(row.get(cluster_key) or row["protein_group_id"])].append(row)
    clusters = sorted(by_cluster)
    point = statistic(rows)
    if len(clusters) < 2:
        return {
            "point": point,
            "ci_low": point,
            "ci_high": point,
            "n_clusters": len(clusters),
            "samples": 0,
        }

    rng = rng_for(seed, len(clusters), samples)
    draws = np.empty(samples, dtype=float)
    for s in range(samples):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        resampled: list[dict[str, Any]] = []
        for index in picked:
            resampled.extend(by_cluster[clusters[int(index)]])
        draws[s] = statistic(resampled)
    return {
        "point": point,
        "ci_low": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_high": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "n_clusters": len(clusters),
        "samples": samples,
    }


def paired_difference(
    rows_a: Sequence[dict[str, Any]],
    rows_b: Sequence[dict[str, Any]],
    *,
    samples: int = 2000,
    seed: Any = "paired",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap CI for a paired within-instance difference (A minus B).

    Used for the minimal-PDB versus normalized-coordinate gap and for rotation
    variants, where the two arms share the same semantic instances.
    """
    a = instance_scores(rows_a)
    b = instance_scores(rows_b)
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n_pairs": 0, "mean_difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    diffs = np.array([a[k] - b[k] for k in shared], dtype=float)
    rng = rng_for(seed, len(shared), samples)
    draws = np.empty(samples, dtype=float)
    for s in range(samples):
        picked = rng.integers(0, len(diffs), size=len(diffs))
        draws[s] = float(diffs[picked].mean())
    return {
        "n_pairs": len(shared),
        "mean_difference": float(diffs.mean()),
        "ci_low": float(np.percentile(draws, 100 * alpha / 2)),
        "ci_high": float(np.percentile(draws, 100 * (1 - alpha / 2))),
        "arm_a_mean": float(mean(a[k] for k in shared)),
        "arm_b_mean": float(mean(b[k] for k in shared)),
    }


# --------------------------------------------------------------------------- #
# Run-to-run consistency
# --------------------------------------------------------------------------- #

def agreement(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pairwise agreement between repeated completions of identical prompts.

    Exact agreement for scalar and categorical answers, Jaccard similarity for
    set answers (section 13).
    """
    by_render: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_render[row["render_id"]].append(row)

    exact: list[float] = []
    jaccard: list[float] = []
    for render_rows in by_render.values():
        if len(render_rows) < 2:
            continue
        is_set = render_rows[0]["answer_schema"] in SET_SCHEMAS
        answers = [_answer_key(r) for r in render_rows]
        for i in range(len(answers)):
            for j in range(i + 1, len(answers)):
                if is_set:
                    jaccard.append(_jaccard(answers[i], answers[j]))
                else:
                    exact.append(1.0 if answers[i] == answers[j] else 0.0)
    return {
        "n_repeated_renders": sum(1 for v in by_render.values() if len(v) > 1),
        "pairwise_exact_agreement": mean(exact) if exact else None,
        "pairwise_jaccard_agreement": mean(jaccard) if jaccard else None,
        "n_exact_pairs": len(exact),
        "n_jaccard_pairs": len(jaccard),
    }


def _answer_key(row: dict[str, Any]) -> Any:
    value = (row.get("parsed_answer") or {}).get("value")
    if isinstance(value, list):
        return frozenset(map(str, value))
    if isinstance(value, dict):
        return frozenset(
            (k, frozenset(map(str, v)) if isinstance(v, list) else str(v)) for k, v in value.items()
        )
    return str(value)


def _jaccard(a: Any, b: Any) -> float:
    if not isinstance(a, frozenset) or not isinstance(b, frozenset):
        return 1.0 if a == b else 0.0
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


# --------------------------------------------------------------------------- #
# Mechanistic episodes
# --------------------------------------------------------------------------- #

OBSERVATION_FIELDS = ("changed_residue", "changed_residues", "gained_pair", "initial_residues")
INTERACTION_FIELDS = (
    "gained_interactions", "gained_interaction", "new_contact", "packing_change",
    "mature_residues", "displaced_entity", "within_protomer", "helix6_displacement",
)


def mechanistic_report(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Observation, interaction and mechanism sub-scores reported separately."""
    mech = [r for r in rows if r["answer_schema"] == "multi_field"]
    if not mech:
        return {}
    levels: dict[str, list[float]] = {"observation": [], "interaction": [], "mechanism": []}
    conditional: list[float] = []
    for row in mech:
        fields = (row.get("detail") or {}).get("fields") or {}
        observation = [v["score"] for k, v in fields.items() if k in OBSERVATION_FIELDS]
        interaction = [v["score"] for k, v in fields.items() if k in INTERACTION_FIELDS]
        mechanism = [v["score"] for k, v in fields.items() if k == "mechanism"]
        if observation:
            levels["observation"].append(mean(observation))
        if interaction:
            levels["interaction"].append(mean(interaction))
        if mechanism:
            levels["mechanism"].append(mean(mechanism))
        if observation and mechanism and mean(observation) >= 1.0:
            conditional.append(mean(mechanism))

    context_only = [r for r in mech if r["representation"] == "context_only"]
    with_coordinates = [r for r in mech if r["representation"] != "context_only"]
    return {
        "n_results": len(mech),
        "observation_score": mean(levels["observation"]) if levels["observation"] else None,
        "interaction_score": mean(levels["interaction"]) if levels["interaction"] else None,
        "mechanism_score": mean(levels["mechanism"]) if levels["mechanism"] else None,
        "mechanism_given_correct_observation": mean(conditional) if conditional else None,
        "context_only_score": micro_score(context_only) if context_only else None,
        "with_coordinates_score": micro_score(with_coordinates) if with_coordinates else None,
        "gain_over_context_only": (
            micro_score(with_coordinates) - micro_score(context_only)
            if context_only and with_coordinates
            else None
        ),
        "reversed_state_order_score": (
            micro_score([r for r in mech if r.get("state_order_seed")]) or None
        ),
    }


def context_only_baseline(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family: what the question alone scores, and what the coordinates add.

    Restricted to instances holding both variants, so the gain is a paired
    contrast rather than two different instance sets compared side by side. A
    family whose gain is near zero is not being answered from the coordinates.
    """
    out: dict[str, dict[str, Any]] = {}
    for family, group in group_by(rows, "question_family").items():
        blind = [r for r in group if r["representation"] == "context_only"]
        if not blind:
            continue
        shared = {r["semantic_instance_id"] for r in blind}
        seeing = [
            r
            for r in group
            if r["representation"] == "minimal_pdb"
            and not r["is_rotation_variant"]
            and r["semantic_instance_id"] in shared
        ]
        if not seeing:
            continue
        blind_score, seeing_score = micro_score(blind), micro_score(seeing)
        out[family] = {
            "n_instances": len(shared),
            "context_only_score": blind_score,
            "with_coordinates_score": seeing_score,
            "gain_over_context_only": seeing_score - blind_score,
        }
    return out


def failure_categories(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(rows)
    return {
        "format_error": sum(1 for r in rows if r["format_error"]),
        "refusal": sum(1 for r in rows if r["refusal"]),
        "truncated": sum(1 for r in rows if r["truncated"]),
        "api_error": sum(1 for r in rows if r["api_error"]),
        "scored_zero": sum(1 for r in rows if r["score"] == 0.0),
        "total": len(rows),
    }
