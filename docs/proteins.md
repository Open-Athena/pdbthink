# The protein pool

Every source structure is listed in `configs/dataset_v1.yaml` and resolved
through `data/manifests/sources_v1.yaml`. The pool has two halves, and they are
there for different reasons.

## Classical structures

Sixty well-characterised proteins — crambin, BPTI, ubiquitin, lysozyme,
carbonic anhydrase, calmodulin, the two-state pairs — chosen because their
geometry is unambiguous and because a curator can sanity-check a gold answer
against textbook knowledge. They are also, without exception, in every training
corpus. That is acceptable for the families whose answers depend on the
*displayed* coordinates (a random rotation makes P03 and G01 unmemorisable) but
it is a real limitation for families whose answer is a property of the molecule
(S08 disulfide partners, S05 fold class).

## FoldBench structures

The second half is drawn from [FoldBench](https://github.com/BEAM-Labs/FoldBench),
whose target lists were assembled for structure prediction: PDB entries released
after 2023-01-01 with low homology to the standard training sets. That is the
property this benchmark wants for a different reason — recency is the cheapest
available proxy for a structure not being memorised.

Five FoldBench lists were read (1552 targets in total):

| list | targets | used |
| --- | --- | --- |
| `monomer_protein.csv` | 334 | 17 |
| `interface_protein_protein.csv` | 279 | 14 |
| `interface_protein_ligand.csv` | 558 | 12 |
| `interface_protein_peptide.csv` | 51 | 6 |
| `interface_protein_dna.csv` | 330 | 0 |

Protein–DNA entries are excluded: nucleic acid is not a supported entity type,
and dropping the DNA would leave a protein whose most interesting features are
contacts with something the model cannot see.

### Screening

Candidates were filtered against the RCSB API before any structure was
downloaded, on criteria that follow from what the benchmark needs rather than
from what FoldBench needs:

- **X-ray only.** Resolution and occupancy are meaningful, and the assembly is
  well defined. No cryo-EM, no NMR ensembles.
- **Resolution ≤ 3.2 Å**, and ≤ 2.0 Å for monomers, where the questions are most
  sensitive to coordinate error (chi1 bins, 2.3 Å disulfide cutoff).
- **Under ~4000 atoms in the deposited entry**, so that the assembly has a chance
  of fitting the context budget after sanitisation.
- **Deduplicated by entry**, since one entry appears in several FoldBench lists.

Seventy-five entries survived. Each was then downloaded, sanitised through the
normal pipeline and *probed*: every generator's `propose` was run against it to
see which families it can actually support. Forty-nine entries supported at
least one constrained family and were added to the config as `fb_<pdbid>` with
tags `[foldbench, post2023]`.

### What they support

Family coverage across the 49 added entries:

| family | entries | family | entries |
| --- | --- | --- | --- |
| G01, G02, G03, S09 | 49 | S05 | 19 |
| N01, S01 | 48 | S06 | 18 |
| P01, P02, P03, S03, S04 | 24 | G04 | 13 |
| | | S07 | 11 |
| | | S08 | 9 |
| | | I01 | 6 |
| | | S02 | 0 |

The unconstrained families gain the most: G01–G03 and S09 can use any structure,
so the FoldBench half nearly doubles their pool and their questions become
questions about structures no model has seen. The constrained families gain
less — I01 needs a genuine two-chain interface that survives cropping, S07 needs
an A.19-eligible metal, and S02 needs a phosphorylated residue, which none of
the 49 has. S02 therefore still runs on the classical pool alone.

`P01`, `P02`, `P03`, `S02`, `S06`, `S07` and `I01` are the families whose
`family_proteins` pool is explicitly listed; the rest draw from the whole set.

### What actually got built

The candidate set is 117 semantic instances over 49 protein groups. Sixty-nine
of them — 59% — sit on a post-2023 FoldBench structure:

| | instances | protein groups |
| --- | --- | --- |
| FoldBench (post-2023) | 69 | 26 |
| classical | 48 | 23 |

Two families fall short of their target and the shortfall is reported by
`validate` rather than hidden: G04 realises 4 of 5, because an unambiguous
single worst clash is rare, and S02 realises 2 of 4, because phosphorylation is
rare in the pool and absent from all 49 FoldBench entries.

## Adding a structure

Add it to `proteins:` in `configs/dataset_v1.yaml`, add it to any explicit
`family_proteins` pool it should serve, run `acquire`, then `build`. If a family
produces nothing from it, `rejections.jsonl` says why. Bump `seed` only when you
intend the selection among equally admissible candidates to change — it will
move every instance, not just the new ones.
