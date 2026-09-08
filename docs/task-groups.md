# Task groups and model results

A reference for what each of the twenty automatic question families asks, plus
the mechanistic episodes, and how the evaluated models perform on them.

Every gold answer is recomputed by an oracle from the exact coordinates the
model is shown, after a random rotation. Nothing about the source — entry ID,
publication, original ligand code, secondary-structure records, B-factors —
reaches a prompt.

## The task groups

`n` is semantic instances in the current candidate set. **Croppable** means the
family's answer is defined over a local neighbourhood, so an excerpt containing
every relevant atom is admissible; an uncroppable family must show the whole
structure and is therefore limited to small proteins.

### Parsing — reading a fixed-column coordinate file

| code | what it asks | answer | n | croppable |
| --- | --- | --- | --- | --- |
| `P01` | List every chain identifier that appears in the structure. | set of chain IDs | 7 | no |
| `P02` | How many protein residues are present in a named chain? | integer | 5 | no |
| `P03` | Report the coordinates of a named atom exactly as they appear. | x, y, z to 3 dp | 5 | no |

`P03` is the only family whose answer depends on the displayed frame — the
rotation changes it — which is why it is excluded from the context-only control.

### Geometry — elementary 3D arithmetic

| code | what it asks | answer | n | croppable |
| --- | --- | --- | --- | --- |
| `G01` | Distance in Ångström between two named atoms. | distance ±0.02 Å | 6 | yes |
| `G02` | Which protein heavy atom is closest to a named atom, excluding itself and anything within two covalent bonds. | atom | 6 | yes |
| `G03` | Which of a supplied candidate list has the smallest minimum heavy-atom distance to a target residue. | residue | 6 | yes |
| `G04` | Which residue pair forms the single most severe steric clash — closer than their atomic radii allow. | residue pair | 4 | no |

### Local structure — geometry turned into structural-biology concepts

| code | what it asks | answer | n | croppable |
| --- | --- | --- | --- | --- |
| `S01` | The salt-bridge partner of a named residue: a Lys/Arg nitrogen within 4.0 Å of an Asp/Glu oxygen. | residue | 6 | yes |
| `S02` | Which residue is phosphorylated — phosphoserine, phosphothreonine or phosphotyrosine. | residue | 2 | no |
| `S03` | Is a named residue buried or solvent-exposed, by relative SASA (≤0.20 buried, ≥0.40 exposed). | 2-way category | 6 | no |
| `S04` | Secondary structure at a named residue. | helix / strand / coil | 6 | no |
| `S06` | Every protein residue with a heavy atom within 4.0 Å of a named ligand. | residue set | 8 | yes |
| `S07` | Every protein residue directly coordinating a named metal ion — N, O or S within the cutoff; carbon never, water-mediated never. | residue set | 7 | yes |
| `S08` | The disulfide partner of a named cysteine: SG atoms within 2.3 Å. | residue | 6 | yes |
| `S09` | The chi1 side-chain rotamer of a named residue. | g+ / t / g- | 6 | yes |

### Global, interface and network

| code | what it asks | answer | n | croppable |
| --- | --- | --- | --- | --- |
| `S05` | Classify a chain's fold as predominantly alpha helical, predominantly beta sheet, or mixed. | 3-way category | 6 | no |
| `I01` | Every residue in one chain with a heavy atom within 4.0 Å of another named chain. | residue set | 7 | no |
| `N01` | The single residue that contacts both of two named anchor residues. | residue | 6 | yes |

### Two-state and mechanism

| code | what it asks | answer | n | croppable |
| --- | --- | --- | --- | --- |
| `T01` | Given the same protein in two superposed states, classify each supplied residue pair as a contact gained or lost in state 2. | two sets | 6 | yes |
| `MECH` | Six curated episodes: connect a local structural change to its functional consequence, scored per field (observation, interaction, mechanism). | episode fields | 6 | yes |

## Per-family results — Kimi K3, the strongest model

`score` is over all renders; `finished` drops responses that hit the output cap;
`floor` is the context-only control, what the question text alone is worth.

| family | score | finished only | truncated | context-only floor |
| --- | --- | --- | --- | --- |
| `G01` | 1.000 | 1.000 | 0/7 | — |
| `G02` | 1.000 | 1.000 | 0/7 | — |
| `G03` | 1.000 | 1.000 | 0/9 | — |
| `N01` | 1.000 | 1.000 | 0/11 | — |
| `P03` | 1.000 | 1.000 | 0/7 | — |
| `S01` | 1.000 | 1.000 | 0/9 | — |
| `S02` | 1.000 | 1.000 | 0/4 | — |
| `T01` | 1.000 | 1.000 | 0/8 | — |
| `S07` | 0.967 | 0.967 | 0/10 | — |
| `S09` | 0.733 | 0.786 | 1/15 | 0.333 |
| `P01` | 0.632 | 0.632 | 0/19 | 0.000 |
| `S08` | 0.625 | 0.714 | 2/16 | 0.000 |
| `P02` | 0.615 | 0.615 | 0/13 | 0.000 |
| `MECH` | 0.454 | 0.757 | 12/30 | 0.522 |
| `S03` | 0.438 | 0.700 | 6/16 | 0.500 |
| `S06` | 0.418 | **0.976** | 8/14 | — |
| `S05` | 0.400 | 0.545 | 4/15 | 0.333 |
| `S04` | 0.250 | 0.500 | 8/16 | 0.167 |
| `G04` | 0.125 | **1.000** | 7/8 | — |
| `I01` | 0.031 | 0.400 | 12/13 | — |

`G04` and `S06` are the clearest cases of the general finding: **1.000 and 0.976
among responses that finished**, against 0.125 and 0.418 as scored. Their low
headline numbers are output budget, not geometry.

With both budget re-runs folded in, the families that were hiding capability
behind the cap recover most of it:

| family | at 32,768 | with both re-runs |
| --- | --- | --- |
| `S06` ligand binding site | 0.418 | **0.968** |
| `MECH` episodes | 0.454 | **0.758** |
| `S05` chain fold class | 0.400 | **0.667** |
| `G04` worst steric clash | 0.125 | **0.625** |
| `I01` chain interface | 0.031 | **0.550** |

`I01` — list every residue of one chain within 4 Å of another — remains the
hardest family even with room to finish, but at 0.550 rather than the 0.031 the
capped run reported.

## Per-model summary

- **coverage** — renders scored out of 247. Gaps are undelivered prompts
  (a credit limit, transient API errors), reported as missing rather than wrong.
- **budget** — `max_output_tokens` for the main run.
- **re-run budget** — the larger budgets used to re-run only the prompts that hit
  the cap. Kimi K3 has two tiers, 64k and 256k, because eleven prompts were cut
  off again at 64k.
- **completion rate** — responses that reached a `FINAL` line rather than being
  cut off. A truncated response scores zero identically to a wrong one.
- **accuracy** — macro average across the twenty families.
- **accuracy | completed** — the same, over responses that finished.
- **accuracy + re-run** — the same, with the higher-budget answers folded in.

| model | coverage | budget | re-run | completion rate | accuracy | accuracy \| completed | accuracy + re-run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Kimi K3** | 247/247 | 32,768 | 64k, 256k | 187/247 = **76%** | 0.684 | **0.830** | **0.804** |
| DeepSeek V4 Flash | 222/247 | 65,536 | — | 177/222 = 80% | 0.615 | 0.756 | — |
| Gemma 4 31B | 155/247 | 32,768 | — | 136/155 = 88% | 0.578 | 0.730 | — |
| MiniMax M3 | 247/247 | 32,768 | 128k | 145/247 = 59% | 0.453 | 0.757 | 0.528 |
| Qwen3.5 9B | 246/247 | 32,768 | 128k | 87/246 = **35%** | 0.294 | 0.583 | 0.337 |
| gpt-oss-120b | 229/247 | 32,768 | 40k | 218/229 = **95%** | 0.257 | 0.270 | 0.259 |
| gpt-oss-20b | 242/247 | 32,768 | 40k | 145/242 = 60% | 0.180 | 0.256 | 0.197 |
| Marin 32B *(base)* | 36/247 | 640 | — | 14/36 = 39% | 0.053 | 0.142 | — |

### How to read it

**Completion rate is the column that explains the others.** Qwen3.5 9B finishes
35% of its responses and its accuracy roughly doubles when the unfinished ones
are dropped, 0.294 to 0.583. gpt-oss-120b finishes 95% and barely moves, 0.257
to 0.270 — its score is a genuine measurement of capability, and it is low.
Two models with similar headline numbers can be failing for opposite reasons.

**The budgets are not equal, and cannot be.** Each model was given the largest
output budget its context window allows beside an 87,500-token prompt. The
gpt-oss models have a 131,072-token context, so 32,768 out and a 40k re-run is
close to all the room there is; Kimi K3 has a million-token context and could be
given far more. This is a real limit on comparability and the reason the budget
re-run is inconclusive for the gpt-oss pair.

Kimi K3 was taken to the end of that road to see where it leads. Its 32,768 cap
was inherited from gpt-oss's ceiling, not chosen for it, and every truncated
response ended at *exactly* the configured limit with `finish_reason: "length"`
— the largest prompt-plus-completion reached 152,204 tokens, **15% of its
window**. Re-running the cut-off prompts twice:

| tier | prompts | before | after | still cut off | macro after |
| --- | --- | --- | --- | --- | --- |
| 32,768 | — | — | — | 60/247 | 0.684 |
| 64k | 47 | 0.000 | 0.615 | 11 | 0.785 |
| 256k | 10 | 0.000 | 0.664 | **0** | **0.804** |

**Most of those truncations were not budget-bound at all.** Seven of the nine
inspected in detail finished *below* the 65,536 cap they had just been
truncated at — 19,510 to 59,558 completion tokens — on the same prompts at
temperature 0.0. The provider's serving is non-deterministic at temperature
zero, which is routine for a mixture-of-experts model under batching, so a
prompt that exhausts 64k on one call can finish in 20k on the next. Only two
genuinely needed more room, one `G04` steric-clash scan running to 97,618
tokens.

That is an argument for the protocol's three-completions-per-prompt rule, which
these runs do not follow. With a single completion per prompt, some of what is
recorded as truncation is run-to-run variance rather than a property of the
question.

**Marin 32B is not comparable at all.** Its 4,096-token context admits no prompt
containing coordinates — the smallest is 7,132 tokens — so its 36 renders are
all context-only controls. The 0.053 is a guessing floor over prompts with no
protein in them.
