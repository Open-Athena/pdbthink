# Operational definitions

Every value that can change a gold label lives in
[`configs/definitions_v1.yaml`](../configs/definitions_v1.yaml) and is recorded
on each instance as `definition_version`. Changing a value requires bumping that
version.

Precedence when requirements conflict:

1. an accepted curator decision,
2. the versioned definitions file,
3. the specification,
4. implementation defaults.

A scientifically material ambiguity is never resolved silently: the candidate is
rejected and the reason is written to `rejections.jsonl`.

## Where each definition is enforced

| appendix | subject | implementation |
| --- | --- | --- |
| A.1 | model and assembly selection | `preprocessing/loader.py` |
| A.2 | minimal processing, rigid transform | `preprocessing/loader.py`, `preprocessing/transform.py` |
| A.3 | alternate locations and occupancy | `loader._resolve_altlocs` |
| A.4 | precision, cutoffs, margins | `configs/definitions_v1.yaml`, every generator |
| A.5 | identifiers | `preprocessing/model.Residue.label` |
| A.6 | heavy atoms | `loader`, `geometry/contacts.py` |
| A.8 | covalent topology, nearest nonbonded | `geometry/topology.py`, `chem.component_bonds` |
| A.9 | residue-residue contact | `geometry/contacts.residue_contacts` |
| A.10 | protein-ligand contact | `geometry/contacts.ligand_contacts` |
| A.11 | salt bridge | `geometry/contacts.salt_bridges` |
| A.12 | phosphorylation | `generators/local.S02` |
| A.13 | SASA and burial | `geometry/sasa.py` |
| A.14–A.16 | secondary structure, fold class | `geometry/dssp.py` |
| A.17 | proline omega | `geometry/rotamer.proline_omega` |
| A.18 | severe clashes | `geometry/contacts.find_clashes` |
| A.19 | metal coordination | `geometry/contacts.metal_coordination` |
| A.20 | disulfides | `geometry/contacts.find_disulfides` |
| A.21–A.22 | chi1 and chi1 change | `geometry/rotamer.py` |
| A.23 | interface contacts | `geometry/contacts.interface_contacts` |
| A.24–A.26 | contact graph, bridges, paths | `geometry/contacts.ContactGraph` |
| A.27–A.30 | two-state mapping and comparison | `geometry/align.py` |
| A.31 | burial on complex formation | `geometry/sasa.isolated_chain_rasa` |
| A.32 | aromatic packing | `chem.AROMATIC_RINGS`, `mechanistic/pipeline.py` |
| A.33 | hydrogen bonds | deferred; disabled in the config |
| A.34 | rejection criteria | `Rejection.criteria_failed` throughout |

## Choices the specification left to the implementation

These are the places where a defensible decision had to be made. Each is
recorded in the configuration rather than buried in code.

**DSSP is implemented in-repo** (`kabsch_sander_v1`) rather than shelling out to
`mkdssp`, so assignments are reproducible from this repository alone. It is
validated two ways: an ideal helix built from standard internal coordinates is
assigned `H` throughout its interior, and on real structures the three-state
assignment agrees with deposited `HELIX`/`SHEET` records 95–100% of the time in
the interior of an annotated span. Boundary residues are exactly where
implementations disagree, so S04 additionally requires the queried residue to sit
at least `min_run_edge_margin` positions inside a run of at least
`min_stable_run` residues of the same class.

**Set-valued questions use a smaller negative margin.** A.4's default 0.20 Å
margin is right for singular questions, where a near-tie makes the answer
genuinely ambiguous. Applied to "list every residue within 4.0 Å", it rejects
essentially every real interface, because interfaces always have a residue near
4.05 Å. Since the prompt states the cutoff and the gold is computed from the same
displayed three-decimal coordinates, the only thing the margin must absorb is
arithmetic slop, so S06 and I01 use `distance.set_question_negative_margin`
(0.05 Å). A.4 explicitly permits this override; it is versioned and recorded.

**Ligand component codes are anonymised, metals and modified residues are not.**
`IXO` becomes `L1`, because the code identifies the source entry. `ZN` stays
`ZN`, because a coordination question is meaningless without the element, and
`SEP` stays `SEP`, because phosphoserine is chemistry rather than provenance.
The mapping is stored privately and `validate` fails if an original code ever
appears in a prompt.

**Chain identifiers are kept when they are single characters.** Assembly
expansion can produce names like `A1`, which do not fit the PDB chain column, so
the whole set is then remapped deterministically and the map is recorded. Two
chains differing only in case would break case-insensitive answer matching, so
P01 rejects such structures outright.

**Insertion codes are excluded in V1.** The identifier format `<chain>:<letter><number>`
cannot express them, so a structure containing one in a retained polymer residue
is rejected rather than silently renumbered.

**AlphaFold entries are gated on pLDDT.** Predicted structures are eligible only
for intrinsic single-chain tasks, and a candidate is rejected if any residue
within `afdb.neighbourhood_radius` of the queried site has pLDDT below
`afdb.min_plddt`. pLDDT is read before B-factors are zeroed and never reaches a
prompt.

**Token budgets are enforced from measured cost.** A rendered minimal-PDB atom
line costs about 41 `cl100k_base` tokens; a normalized-coordinate row costs about
22. Families that may not be cropped therefore refuse structures above
`MAX_UNCROPPABLE_ATOMS`, and the builder still checks the exact count per render.
The measured 2× difference is itself a finding: the representation control is
comparing formats with very different token costs for identical information.
