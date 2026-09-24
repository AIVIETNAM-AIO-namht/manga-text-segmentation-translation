# Research: Text Removal & Image Inpainting

**Date**: 2026-09-24 | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

All unknowns resolved. No NEEDS CLARIFICATION remains. Seven decisions, in Decision / Rationale /
Alternatives-considered form.

## R1 — `classical_baseline` is one identity, and which mask it means

**Decision**: FR-026's tree has exactly four method directories and one of them is `classical_baseline`.
`configs/inpainting.json` resolves each identity to the place its masks come from. For the three
deep-learning identities that place is `deliverables/<run_id>/<method>/masks/` (Spec 2's admitted layout,
FR-043). For `classical_baseline` it is a **single named classical method from Spec 1's frozen run** —
default `adaptive` — read from `outputs/segmentation/default/adaptive/`. The identity, the method it
resolves to and the reason are recorded in the run record, so an inpainted page can always be traced to
the exact mask that produced it.

**Rationale**: Spec 1 emitted **six** classical methods, not one. `outputs/segmentation/default/` holds
`adaptive`, `components`, `edges`, `morphology`, `mser` and `otsu`; `metrics.csv` and `summaries.json`
each carry six rows; and Spec 2's `assemble_comparison` (`src/manga_text_seg/benchmark.py`) emits one row
per classical method, i.e. six. Meanwhile every *downstream* feature speaks of a single baseline:
Spec 4's FR-009 names the method set as "(`classical_baseline`, `manga_text_segmentation`,
`comic_text_detector`, `unetpp_efficientnetv2`)"; Spec 5's Assumptions say "the four method names are
fixed as in Spec 2"; Spec 2's own Comparison Summary entity says "one row per method including the
classical baseline"; `README.md:9` describes "Classical baseline" as one thing. Spec 3 is where the
six-to-one collapse has to happen, because FR-026's tree gives `classical_baseline/` a single
`{telea, ns}` pair with no room for a method level beneath it.

The collapse is made **by configuration, not by a metric**. `summaries.json` publishes the frozen
ranking — `adaptive` mean IoU 0.0711 is the highest of the six (`components` 0.0563, `morphology` 0.0502,
`mser` 0.0435, `edges` 0.0350, `otsu` 0.0081) — so `adaptive` is the documented default. But the choice is
*pinned in config*, never recomputed at run time. Selecting the best-performing classical method
dynamically would make a run's inputs depend on metrics computed during that run: any later change to the
metric code would silently switch which mask was inpainted, and the run ID would no longer identify its own
inputs. That is the same class of failure FR-008 forbids on the intake side ("Neither case MAY be resolved
by similarity or guesswork") and FR-023 forbids outright (determinism, FR-023/SC-012).

Two consequences the plan states rather than hides: (a) the `classical_baseline` in an inpainting
comparison is a **labelled representative**, not a claim that Spec 1's six methods are equivalent —
Spec 2's report still shows all six, and that is unchanged; (b) whichever method is chosen, the mask
processing and the inpaint radius are identical across all four identities (FR-017, FR-021) — the choice
selects *which mask*, never *how it is processed*.

**Alternatives considered**:
- *Process all six classical methods under `classical_baseline/`* — rejected: FR-026's tree is
  `classical_baseline/{telea,ns}` with no intermediate level, and adding one would make Spec 3's output
  tree disagree with the tree three other features are being written against. It would also multiply the
  classical side of the benchmark by six while the DL side has one identity each, which is not a
  comparison.
- *Pick the best classical method by IoU at run time* — rejected on determinism grounds, above. It also
  makes the feature's output depend on Spec 2's metric module, which FR-030 forbids in the other
  direction (inpainting outputs must not feed segmentation metrics).
- *Treat `classical_baseline` as a group label with no mask of its own* — rejected: FR-026 gives it a
  directory containing actual outputs, and Spec 4's FR-012 selects an inpainting source by method; an
  identity that cannot be processed cannot be selected.
- *Average or union the six classical masks into one* — rejected: it would fabricate a mask no method
  produced, and the metadata record (FR-025) has a `segmentation method` field that would then be a lie.

## R2 — N for FR-034's selection rules

**Decision**: **N = 5**, the spec's stated initial default, made configurable as `selection.n` in
`configs/inpainting.json`. N is fixed **per selection rule per segmentation method**, exactly as FR-034
writes it. If a rule yields fewer than N candidates for a method, the shortfall is recorded in the
selection report — the list is never padded with samples that do not satisfy the rule.

**Rationale**: FR-034 delegates the number to planning and supplies N=5 as the starting point; nothing in
the spec, the repository or the frozen results argues for a different one, and deviating without cause
would be an unrequested change. The volume is the real constraint: five rules (highest IoU/F1, lowest
IoU/F1, most false positives, most false negatives, manual artifact flags) × four method identities ×
5 samples = up to 100 five-panel boards (FR-033), and FR-032/FR-035 put a **human** in front of every one
of them to fill in seven criteria plus a written observation. 100 is already a substantial review; 200
(N=10) is past what a reviewer completes, and N=3 is too few for a rule like "most false negatives" to
show any variety.

Note the fifth rule: "manually flagged artifact cases" has no computed candidate set, so its samples come
from a human-supplied list in the selection configuration. Its count is therefore *at most* N, and the
report says so rather than inventing flags.

**Alternatives considered**:
- *N = 3* — rejected: the point of the boards is to show the failure modes side by side across methods;
  three samples per rule does not distinguish a systematic behaviour from a coincidence.
- *N = 10* — rejected: doubles the human review load for no stated benefit, and the spec's own default is
  half that.
- *Make N adaptive to dataset size* — rejected as speculative generality (YAGNI): the dataset is fixed at
  390 pages and the spec asks for a fixed count.

## R3 — What determinism forbids (FR-023, SC-012)

**Decision**: SC-012 requires byte-identical outputs across two consecutive runs on identical inputs and
configuration. Three things are therefore fixed: (a) **no wall-clock timestamp appears in any per-sample
metadata record** — the only time-identifying field is the run ID, following Spec 2's page-list precedent
of `generated_at_run` rather than wall-clock; (b) **every iteration over pages, method identities and
algorithms follows a sorted, total order** — the page list's `(manga, stem)` order and the literal order
of the configured identities — never a `set` or a `dict` whose order is incidental; (c) **per-sample
processing time is recorded but excluded from the byte-comparison**, because it is a measurement of the
machine, not of the input. FR-022 requires that timing to be recorded, and FR-025 requires it in the
metadata record, so it stays in the record; the determinism check compares the *artifacts* (masks, page
images, the structural fields of the metadata) and explicitly not the timing field.

**Rationale**: SC-012 is stated without qualification, and the three items above are the only sources of
run-to-run variation in an otherwise deterministic pipeline — `cv2.inpaint`, `cv2.dilate` and PNG encoding
are deterministic functions of their inputs and parameters. Stating the timing carve-out explicitly is
what keeps FR-022 and SC-012 from appearing to contradict each other; leaving it implicit would produce a
determinism test that fails for the wrong reason, or a feature that quietly drops a required field.

**Alternatives considered**:
- *Drop per-sample timing to satisfy SC-012* — rejected: FR-022 requires it, and FR-025 lists it in the
  metadata record. Dropping a required field to make a test pass is the wrong trade.
- *Round timings to a fixed grid so they compare equal* — rejected: it would corrupt the aggregate FR-022
  asks for (mean processing time per image, per method × algorithm) to satisfy a test that can simply
  exclude the field.
- *Include a wall-clock timestamp and exclude it from the comparison* — rejected: it invites a diff-based
  determinism check that has to know which fields are decorative, which is exactly the kind of implicit
  rule that breaks later. The run ID carries the time information a human needs.

## R4 — Output format, and what "lossless by default" fixes

**Decision**: The default output format is **PNG** for both the inpainted page and the masks, and the
effective format is recorded per sample. FR-028's requirement is that the default be lossless; PNG is the
format the project already writes masks in (`imaging.save_mask`) and adds no dependency.

**Rationale**: FR-028 makes losslessness a default rather than a constraint, so the format must be
configurable (FR-038) — but the comparison this feature exists to enable is a comparison of *pixels*, and a
lossy default would perturb the very thing being compared, differently on each re-encode. Choosing the
format the repository already uses also means the page images and the masks round-trip through one code
path.

**Alternatives considered**:
- *JPEG default* — rejected: lossy, and it would make SC-012's byte-identical requirement unsatisfiable in
  spirit even where the encoder happens to be deterministic.
- *Lossless WebP* — rejected: no advantage over PNG here and it is not a format the project already
  handles; adding an encoder path for no gain is the kind of thing YAGNI exists to prevent.
- *Uncompressed BMP/TIFF* — rejected: 3,120 pages of uncompressed raster is a large amount of disk for no
  benefit; PNG's lossless compression is free.

## R5 — Where intake reads from, and where it does not

**Decision**: The intake root is the **admitted** layout only: `deliverables/<run_id>/<method>/masks/` plus
the sidecars in `deliverables/<run_id>/<method>/metadata/`, and for the classical identity Spec 1's own
`outputs/segmentation/default/<method>/`. The staging area `notebooks/deliverables/` is **not** an intake
path, and `benchmark/availability.json` is **not** consulted.

**Rationale**: `deliverables/<run_id>/<method>/` is what FR-043 promised a downstream consumer would find,
and what `deliverables/README.md` documents as the mapping Spec 3 consumes. It is also the only layout
whose contents have already passed FR-058's receipt validation — `notebooks/deliverables/` is pre-admission
and may hold refused results, as REISSUE.md's three hand-offs currently do. Reading the staging area would
mean scoring material the admission gate rejected, which inverts the whole point of having a gate.

Not consulting `availability.json` is the same fact REISSUE.md §7 records: availability answers "can *this
machine* run the model", and Spec 3 never runs a model — inference happened on the runner's machine, and
the result is already on disk. `admit` does not import `availability` either, for the same reason. A
feature that refused to inpaint a mask because *its own* machine lacks a checkpoint would refuse exactly
the results that are admissible.

**Alternatives considered**:
- *Read `notebooks/deliverables/` as a convenience* — rejected: it bypasses the admission gate and would
  process the three refused hand-offs sitting there today.
- *Require the admitted `metrics/per-page.csv` and refuse without it* — rejected: it is this project's own
  computed output, not part of the hand-off; the intake re-verifies the masks rather than depending on a
  file it wrote itself.
- *Gate on `availability.json` for the DL identities* — rejected above; it would make the feature's
  behaviour depend on the local machine's checkpoint inventory, which is irrelevant to it.

## R6 — The output tree, and why `outputs/` staying gitignored is correct

**Decision**: Inpainted pages, processed masks, metadata records and the error report are written under
`outputs/inpainting/<run_id>/`, which is gitignored (`.gitignore:23` is `outputs/`). The asymmetry with
the committed `deliverables/` is deliberate and is recorded here rather than left to look like an
oversight.

**Rationale**: `outputs/` holds the project's **own regenerable product**: Spec 1's masks and Spec 3's
inpainted pages are both derivable from committed inputs by a committed command (`manga-text-seg run` and
the new inpaint subcommands), on a CPU, with no network and no checkpoint. `deliverables/` holds the
**runners' submitted evidence**, which this project cannot regenerate at all — delete it and it is gone,
because the checkpoints and the pinned environments live on machines this repository does not control.
So the rule is not "outputs are unimportant" but "committed iff not regenerable here". FR-027 makes the
untracked tree safe: every output is namespaced by run ID and a run MUST NOT overwrite a previous run's
outputs, so a fresh clone re-derives into a new run ID rather than colliding with one it cannot see.

The size argument agrees: 3,120 inpainted pages at roughly the size of the source pages is on the order of
a gigabyte of committed binary, regenerable in minutes.

**Alternatives considered**:
- *Commit `outputs/inpainting/`* — rejected: it is regenerable, it is large, and committing it would put
  binary artifacts in the tree that a clone can reproduce, which is the definition of the thing
  `.gitignore` exists to prevent. It would also weaken SC-013's reviewability by burying the tree that
  must contain zero weights among gigabytes of legitimate binaries.
- *Write into `deliverables/<run_id>/<method>/`* — rejected: that tree is the runners' hand-off layout, and
  FR-030 forbids using inpainting outputs as segmentation results. Writing inpainted pages into the
  admitted-result tree would blur the two and put Spec 3's product under a path that means "submitted by a
  runner".
- *Write beside the source pages under `outputs/segmentation/`* — rejected: FR-024 requires ablation and
  main-benchmark outputs to be separable, and mixing Spec 3's outputs into Spec 1's run directory would
  make both harder to reason about.

## R7 — The contract Spec 4 and Spec 5 will read

**Decision**: `contracts/` fixes the inpainted-output contract normatively, because two features already
describe what they expect from it. Specifically it pins: (a) the addressable path
`outputs/inpainting/<run_id>/<method>/<algorithm>/<manga>/<page_id>.png` plus its metadata sidecar, so
FR-029's consumer — which knows only the run ID, the segmentation method and the `image_id` — can locate
and interpret a page with nothing else; (b) the field names of the per-sample metadata record, which are
the union of FR-025's list and FR-018's mask-processing record; (c) that the **raw prediction mask copy**
is byte-identical to Spec 2's admitted mask for that page, so "the raw copy" and "Spec 2's output" are the
same object and not two files that might drift; (d) that the **post-dilation mask** — Spec 5's "processed
mask" — is the mask actually handed to `cv2.inpaint`, written as its own artifact beside the raw copy.

**Rationale**: FR-029's requirement is precise and easy to satisfy by accident in a way that breaks
downstream: it says a consumer knowing only run ID, method and `image_id` must locate *and correctly
interpret* the page "without run-specific knowledge beyond that". A run-level index file would satisfy
"locate" only for a consumer willing to read the index, which is run-specific knowledge. A deterministic
path does it with no index at all. Items (c) and (d) come from Spec 5's Assumptions, which already assume
Spec 3 exports both masks and distinguishes them; writing that down now is what makes Spec 5's assumption
true rather than hopeful. Spec 4's FR-012 selects "a Spec 3 run ID and its algorithm" — the same two keys
that address the path, so the contract and the selector agree by construction.

**Alternatives considered**:
- *A run-level manifest as the only addressing mechanism* — rejected: it makes FR-029's "without
  run-specific knowledge" false, and it makes a partially-copied run directory unusable.
- *Leave the two mask artifacts undifferentiated* — rejected: Spec 5 assumes they are distinct and
  labelled, and the qualitative comparison is about the processed mask while the provenance is about the
  raw one; conflating them would make a board unable to say which it is showing.
- *Have Spec 4 read Spec 2's masks directly and skip Spec 3's raw copy* — rejected: Spec 4's FR-012
  explicitly selects an inpainting source, so it needs a Spec 3 run to point at; the raw copy is what lets
  it confirm which prediction the inpainted page came from.
