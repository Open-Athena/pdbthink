# Original benchmark revision 1.1

Revision 1.1 corrects question and scoring defects found while auditing completed,
incorrect model responses on the original PDBThink benchmark. Rebuild the dataset
before evaluating this revision. Historical scores and released coordinate-task
bundles retain their original versions.

| Component | Version |
| --- | --- |
| Original dataset configuration | 1.1.0 |
| Prompt library | v6 |
| Mechanistic episodes | 2.0.0 |
| Scorer | 1.2.0 |
| Geometric definitions | v1.0.0, unchanged |

## Corrections

Protein residue identifiers now use the parent amino-acid letter for D-amino
acids and the identifier letter N for SNN. Coordinate blocks retain their
component names and include an explicit component-to-letter legend. SNN receives
an identity mapping only; its cyclic structure does not inherit ASN's atom-name
or bond topology. Unknown protein components still use X.

The I01 interface question explicitly excludes non-polymer entities in both
chains. Its previous wording said every residue, although the oracle counted
only protein residues. This matters when ligands share a protein chain ID.

Mechanistic questions now state the geometric selection rules needed to answer
the audited fields:

- M1 (M2 receptor allosteric modulation) selects the largest circular chi1
  change among residues contacting the specified allosteric ligand. Its gained-contact field uses the stated 4 Å
  cutoff in both states, removing an unstated 4.5 Å non-contact threshold.
- M2 (CB1 activation) and M3 (A2A receptor–G-protein coupling) name candidate
  residues and ask which exceed the chi1-change threshold. The former request for the “clearest” or “most important”
  rearrangement did not define a reproducible selection rule. M2's packing
  comparison explicitly uses side-chain heavy atoms. M3 gives the chain and
  author-number range for its displacement measurement.
- M3 accepts every receptor–G-protein contact satisfying the question's cutoff.
- M4 (beta2-adrenergic receptor–G-protein coupling) asks for all displayed
  G-protein residues contacting the receptor and computes the complete sets. It previously scored a selected subset of
  paper-derived examples without specifying that restriction.
- M6 (TRPV1 activation) defines a gained Arg–Glu charged-atom contact and accepts
  qualifying pairs in every displayed protomer. The accompanying within-protomer question rejects
  an episode if some qualifying pairs lie within one protomer and others span protomers.

Every mechanistic rendering recomputes its geometric gold from its rotated,
rounded coordinates. An episode whose stored gold differs between the primary
and alternate rotation, including a rounded numerical value or accepted contact set, is
omitted from that build with a rejection record. Reversed-state variants swap
state references in both context and questions; their answers retain the meaning of those rewritten questions.
Single-pair scoring supports a canonical answer plus explicitly enumerated valid
alternatives, including within a multi-field answer.

Mechanism multiple-choice letters remain curated interpretations of the supplied
experimental context. Coordinate calculations do not independently establish
causality. These corrections do not constitute an independent scientific review
of every question in the benchmark.

## Rebuild and compare

Use the build, validation and mock-gold commands in [AGENTS.md](../AGENTS.md),
with `configs/dataset_v1.yaml` for the full original benchmark. Preserve the old
bundle and record the new prompt, chemistry and scorer fingerprints with results.

Changed wording requires new model responses. Do not relabel old results as
revision 1.1 results after changing their answer key. A retrospective scorer-only
comparison is valid only for unchanged questions whose accepted answer set was
incomplete, and must be reported separately.

Regression tests use synthetic structures. Actual rendered prompts, answer keys
and per-case audit evidence remain outside public Git, following the
[contamination policy](contamination.md).
