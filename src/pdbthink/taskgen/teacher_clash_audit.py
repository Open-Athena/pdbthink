"""Recompute the documented clash case without using the geometry oracle."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pdbthink.chem import component_bonds, three_to_one


def audit(task: dict) -> dict:
    """Isolate the effect of an unstated SG–SG exclusion in one frozen prompt."""
    if not __debug__:
        raise RuntimeError("This audit requires Python assertions; do not use python -O")
    assert task["path"] == "pdbthink-g04-908a6b9e20e6ee9b26bd"
    contract = json.loads(task["gold_json"])
    assert contract["definition_version"] == "v1.0.0"
    # Freeze the declared v1 constants for this historical case; this diagnostic
    # must remain independent of changes to the live geometry implementation.
    atoms = []
    residues = defaultdict(dict)
    for line in task["user_prompt"].splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        assert line.startswith("ATOM  ")
        key = (line[21], int(line[22:26]), line[26])
        atom = {
            "name": line[12:16].strip(),
            "resname": line[17:20].strip(),
            "residue": key,
            "xyz": [float(line[i : i + 8]) for i in (30, 38, 46)],
            "element": line[76:78].strip(),
            "record": line,
        }
        residues[key][atom["name"]] = len(atoms)
        atoms.append(atom)
    assert len(atoms) == 536
    assert len(atoms) == sum(len(names) for names in residues.values())
    assert sorted(set(a["element"] for a in atoms)) == ["C", "N", "O", "S"]
    bonds = [set() for _ in atoms]

    def link(i: int, j: int) -> None:
        bonds[i].add(j)
        bonds[j].add(i)

    for key, names in residues.items():
        for a, b in component_bonds(atoms[next(iter(names.values()))]["resname"]):
            if a in names and b in names:
                link(names[a], names[b])
    ordered = list(residues)
    assert [key[1] for key in ordered] == list(range(1, 68))
    for first, second in zip(ordered, ordered[1:]):
        a, b = residues[first].get("C"), residues[second].get("N")
        if a is not None and b is not None and math.dist(atoms[a]["xyz"], atoms[b]["xyz"]) <= 2.0:
            link(a, b)
    sgs = [i for i, a in enumerate(atoms) if a["resname"] == "CYS" and a["name"] == "SG"]
    candidates = sorted(
        (math.dist(atoms[i]["xyz"], atoms[j]["xyz"]), i, j) for n, i in enumerate(sgs) for j in sgs[n + 1 :]
    )
    used = set()
    disulfides = []
    for d, i, j in candidates:
        if d <= 2.30 and i not in used and j not in used:
            link(i, j)
            used.update((i, j))
            disulfides.append((i, j, d))
    two_hops = []
    for i in range(len(atoms)):
        excluded = {i} | bonds[i]
        for j in bonds[i]:
            excluded |= bonds[j]
        two_hops.append(excluded)
    radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}

    def label(a: dict) -> str:
        return f"{a['residue'][0]}:{three_to_one(a['resname'])}{a['residue'][1]}"

    def ranking(exclude_sg_sg: bool) -> list[dict]:
        best = {}
        for i, a in enumerate(atoms):
            for j in range(i + 1, len(atoms)):
                b = atoms[j]
                if a["residue"] == b["residue"] or j in two_hops[i]:
                    continue
                if exclude_sg_sg and a["name"] == b["name"] == "SG" and a["element"] == b["element"] == "S":
                    continue
                d = math.dist(a["xyz"], b["xyz"])
                overlap = radii[a["element"]] + radii[b["element"]] - 0.40 - d
                if overlap <= 0:
                    continue
                pair = "--".join(sorted((label(a), label(b))))
                if pair not in best or overlap > best[pair]["overlap_after_tolerance"]:
                    best[pair] = {
                        "pair": pair,
                        "distance": d,
                        "overlap_after_tolerance": overlap,
                        "records": [a["record"], b["record"]],
                    }
        return sorted(best.values(), key=lambda r: (-r["overlap_after_tolerance"], r["pair"]))

    native = ranking(True)
    including_sg = ranking(False)
    gold = contract["gold_answer"]["value"]
    assert native[0]["pair"] == gold
    assert including_sg[0]["pair"] == "A:C11--A:C50"
    audit = {
        "task_id": task["path"],
        "prompt_sha256": task["prompt_sha256"],
        "family": "G04",
        "definition_version": contract["definition_version"],
        "method": (
            "Independent fixed-width parsing of displayed PDB, math.dist for all cross-residue atom pairs, "
            "independent covalent graph construction and two-hop exclusions; uses the versioned CCD-lite "
            "bond dictionary and declared radii, without calling a geometry oracle."
        ),
        "atoms": len(atoms),
        "residues": len(residues),
        "radii": radii,
        "overlap_tolerance": 0.40,
        "gold_answer": gold,
        "native_rule_ranking_top5": native[:5],
        "ranking_without_the_SG_SG_exclusion_top5": including_sg[:5],
        "native_margin_over_runner_up": native[0]["overlap_after_tolerance"]
        - native[1]["overlap_after_tolerance"],
        "teacher_attempt": 10,
        "teacher_answer": "FINAL: A:C11--A:C50",
        "question": task["user_prompt"].split("Question: ", 1)[1],
        "finding_type": "unstated_clash_exclusion",
        "native_reward_unchanged": True,
        "finding": (
            "The native gold is reproduced using the declared clash exclusions. The teacher-selected "
            "cysteine SG-SG pair has the greatest overlap if the blanket SG-SG exclusion is removed. "
            "Its 2.34967 Angstrom distance is outside the separate 2.30 Angstrom disulfide-detection "
            "cutoff. The displayed question does not explicitly state the SG-SG exclusion. This case "
            "limits how strongly a native failure can be interpreted as an error in distance arithmetic."
        ),
        "scope": (
            "One deliberately selected exhausted case; no frequency estimate and no change to native "
            "rewards or prompts."
        ),
    }
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.task_json.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "task_id": result["task_id"],
                "gold_answer": result["gold_answer"],
                "native_margin_over_runner_up": result["native_margin_over_runner_up"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
