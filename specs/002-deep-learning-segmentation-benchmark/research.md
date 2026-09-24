# Research: Deep Learning Segmentation Benchmark

**Date**: 2026-09-19 | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

All unknowns resolved. No NEEDS CLARIFICATION remains.

## R1 — What the distributable page list is, and why it is not the manifest

**Decision**: The export is a new derived file `benchmark/page-list.json`, written by a new `export`
subcommand, carrying exactly: (a) `page_list_identity` — a sha256 computed over the canonical serialisation
of the list itself; (b) the ordered list of page identifiers `{manga, stem, image_id}`; (c) for each page a
reference to the raw image by relative path **inside the bundle** plus its `input_image_identity` (sha256 of
the exact image bytes handed out); (d) the aligned target size `[1654, 1170]` every mask must match; (e) the
alignment convention name and the mask value convention. It carries **no** `gt_root`, **no** `mask_path`,
**no** ground-truth reference of any kind, and **no** metric.

**Rationale**: FR-054 requires the list to be self-contained and consumable without this project's data
tree; FR-054a requires the input images to be distributable and every method to run on the same bytes the
classical baseline ran on. The Spec 1 manifest cannot serve: `Manifest.to_dict()` in
`src/manga_text_seg/manifest.py` emits `gt_root`, `raw_root` and a per-pair `mask_path`, i.e. it *names the
answers*. Handing a runner the manifest would hand them the GT location, which is precisely what ONBOARDING
§3's ground-truth subsection forbids. The list is therefore derived from the manifest, never a copy of it.
Computing the identity over the list's own canonical bytes means a runner who received the list days ago can
prove *which* list their result came from, and the project can refuse a result built from a stale or
self-built one (FR-058).

**Alternatives considered**:
- *Ship the manifest with GT fields stripped* — rejected: the file's identity would then depend on a
  subtraction step applied by each runner, and the project could not distinguish "manifest with GT removed
  by us" from "manifest with GT removed by them". A distinct artefact with a distinct identity is
  verifiable; a redacted shared artefact is not.
- *Ship a page list of ids only* — rejected: page identifiers alone align by name, which is exactly the
  stale-list failure the spec's edge cases call out. The identity must bind the *list instance*, not the
  page names (FR-054's "same list instance").
- *Have runners build their own list by filename matching* — explicitly forbidden: ONBOARDING §3 states this
  violates FR-001/002/054/054a and the result is refused even if it produces the correct 390 pages.

## R2 — Input-image identity, and how the images travel

**Decision**: The 390 paired raw images are bundled into a single archive produced by `export` under
`benchmark/dist/`, with `image-identity.json` (committed) recording each page's sha256 over the raw file
bytes. Receipt validation recomputes the hash of the runner's returned metadata's claimed identity against
the committed record; the runner's own copy of the images is never trusted by assertion, only by hash.

**Rationale**: FR-054a's requirement is that every method runs on the same bytes the classical baseline
ran on, and that per-page identity be strong enough to verify on receipt. Measured sizes make this
tractable: the 390 paired raw images total 144,364,052 bytes (137.7 MiB) even though the full released
image tree is 3.2 GB — so the bundle is ~138 MiB, not 3.2 GB, and the largest single file is 627,003 bytes.
The bundle is a distribution convenience; the hash record is the contract. If a runner obtains the images by
any other route, the identity check still decides whether their result is admissible.

**Alternatives considered**:
- *Commit the images into the repository* — rejected: ~138 MiB of binary in git history, against the
  existing `.gitignore`'s `data/` and `*.zip` rules, and it would duplicate a dataset the project already
  holds read-only. The committed artefact is the identity record.
- *Reference the upstream dataset release instead of shipping bytes* — rejected: the upstream release is
  3.2 GB and contains 60 GT masks with no images at all; pointing a runner at it would make "the same bytes"
  an assumption rather than a check.
- *Per-page hashing only, no bundle* — kept as the contract; the bundle is just the delivery mechanism.

## R3 — Receipt validation order, and why size is checked before metrics

**Decision**: `receipt.py` validates in exactly this order, and refuses at the first failure with the
specific reason named and **nothing** written into the run: (1) `page_list_identity` present and equal to
the committed export's identity; (2) per-page `input_image_identity` matches the committed record;
(3) mask exists, is single-channel, and its size equals the aligned page size `1654×1170`;
(4) every pixel value is in `{0, 255}`; (5) provenance is complete per R6; (6) for Method A,
`fold_attribution` is present and consistent with the derived book→fold map. Only after all six pass are
masks copied into `deliverables/<run_id>/<method>/` and metrics computed.

**Rationale**: FR-058 fixes this order and forbids silent correction, resizing, re-thresholding or dropping.
The ordering is not stylistic: `metrics.compute_metrics` in `src/manga_text_seg/metrics.py` calls
`_as_binary` on both inputs and then raises `MetricError(f"Shape mismatch: gt {gt.shape} vs prediction
{prediction.shape}; align first")` — it does not resize. A size check that ran *after* metric computation
would therefore surface as a crash rather than as the named refusal FR-058 requires, and worse, a
well-meaning fix would be to resize, which is the silent correction the requirement bans. A refused result
must leave the run reportable over already-admitted results (FR-059, SC-014), so refusal writes nothing.

**Alternatives considered**:
- *Validate provenance first* — rejected: provenance completeness is the cheapest check but the least
  informative failure; a runner whose identity fields are wrong needs to be told that, not that their
  licence field is empty. FR-058's order also happens to run cheap-to-expensive from identity outward.
- *Auto-resize a mask whose size is wrong* — forbidden by FR-058 and by the edge case on internal
  resolution mismatch: a mask left at network resolution is a defect, not a result.
- *Accept and flag, scoring with a caveat* — rejected: FR-058 admits or refuses; SC-014 requires the refusal
  to be recorded with the specific missing field.

## R4 — Method A's leave-one-fold-out attribution (FR-013a)

**Decision**: `adapters/manga_text_segmentation.py` derives the book→fold map from upstream's published
split: the 45 book names sorted and split into 5 folds under the published seed 42 with shuffle enabled,
reproducing the training code's split exactly. Each page is inferred by the fold whose *validation* split
contained that page's book — i.e. the one checkpoint that never trained on it. The derivation is verified
against the published fold checkpoints' own book lists before any page is inferred; if it cannot be
reproduced, Method A is reported `unavailable` with that reason and no result is produced. The
`fold_attribution` field is written per page and re-checked on receipt (R3 step 6). All five checkpoints are
verified present at availability-check time, not at inference time.

**Rationale**: Every book was held out from exactly one of the five folds and trained on by the other four,
so for any page at most one released checkpoint is uncontaminated by it, and which one is computable from
the published seed. Using one checkpoint for all 390 pages is not a weaker result but an invalid one —
ONBOARDING §4 states this directly. De-contaminating Method A is also what makes the asymmetry inside the
DL group explicit: A is de-contaminated, C is fully contaminated (its published training set is
set-identical to this GT), and B's extent is unknowable. SC-011 requires that disclosure next to every
method's scores.

**Alternatives considered**:
- *Use one checkpoint for all pages* — invalid per FR-013a; also maximally contaminated for 4/5 of pages.
- *Approximate the mapping by page similarity or by a re-derived split* — rejected: FR-013a says unmappable
  means inadmissible, not approximated. Approximating would silently mix contaminated and clean scores in
  one mean.
- *Run all five checkpoints on every page and report the best* — rejected: that is per-page selection, which
  FR-026/FR-030 forbid, and it inflates the metric by construction.

## R5 — Padding multiples, threshold, channel order and the lossless-collapse check

**Decision**: Padding, threshold and channel order are **per-method recorded configuration**, inverted and
disclosed separately, and the effective threshold is verified against the produced mask rather than read
from configuration:

- **Padding (FR-024a)**: the aligned page size is a multiple of 8 (1654→1656, 1170→1176 — the GT's own
  pad-to-multiple-of-8, verified across all 450 masks) but **not** of 32. Method B letterboxes to a
  1024×1024 stride-64 canvas; Method C zero-pads to a multiple of 32; Method A pads to a multiple of 8.
  Each method's multiple is recorded and inverted separately. Collapsing them into one rule misplaces the
  mask by a few pixels on every page, which on a dataset with mean stroke width 3.06–14.50 px is a large
  fraction of the signal.
- **Threshold (FR-026/FR-030)**: Method B accepts a threshold parameter its code ignores in favour of a
  hardcoded integer cutoff around 0.235 on the sigmoid output. Configuring a value the code does not honour
  would put a lie in the metadata, so the run records the threshold *actually applied*, and the adapter
  verifies it against the produced mask (e.g. re-thresholding the returned probability map at the recorded
  value reproduces the mask). Method A's 0.5 sigmoid threshold is fixed and not exposed; Method C's default
  is 0.5. Threshold and morphology are method-level configuration fixed for the whole run, never tuned per
  page.
- **Channel order**: at least one method has a suspected red/blue reversal between training-time and
  inference-time preprocessing. It degrades accuracy **without raising an error**, so the order actually used
  is recorded; a suspected reversal is reported as a recorded deviation, not silently corrected.
- **Lossless collapse (FR-024b)**: Method A's training label space distinguishes easy text (class 1) and
  hard text (class 2) and reserves classes 3–5 as ignore; upstream evaluation counts 1 and 2 as text and
  excludes 3 from both hits and misses. Because this benchmark's GT is the post-processed binary variant,
  the collapse to the FR-004 binary convention is expected to be lossless — but the adapter **verifies** it
  per page (no GT pixel is left unrepresented and no non-text class is admitted as text) and records the
  collapse rule in run metadata. Verification, not assumption, is what FR-024b requires.

**Rationale**: Every one of these four is a silent-failure vector: each produces a mask of the correct
shape, dtype and value set while being wrong. Since FR-058's value-set and size checks cannot catch any of
them, the only defence is that the adapter records what it did and the run metadata is complete enough for a
reader to know which transformation was applied. The measured dataset properties — text occupying
0.86%–11.9% of page area, 69–724 components per page, IoU as low as 0.446 between a mask and its own
single-pixel 3×3 erosion — mean one-pixel boundary disagreement is heavily penalised, which is why
per-page tuning and undisclosed post-processing are forbidden rather than merely discouraged.

**Alternatives considered**:
- *Read the threshold from config and record it* — rejected: that records intent, not behaviour. The edge
  case names this explicitly as a silent lie.
- *Normalise all methods to a common internal resolution* — rejected: it would require resizing masks back
  and inventing geometry the methods did not use, and FR-024/FR-025 require each method's own mapping to be
  inverted and recorded.
- *Assume the class collapse is lossless because the GT is post-processed* — rejected by FR-024b's explicit
  "verify — not assume" wording.
- *Apply one padding inverse for all methods* — rejected by FR-024a; 8 and 32 do not commute on a page that
  is a multiple of 8 but not of 32.

## R6 — Provenance completeness, and positive checkpoint-load evidence

**Decision**: `provenance.py` defines the per-page record as ONBOARDING §6's schema and machine-checks it
for completeness before admission. Required per page: `image_id`, `page_list_identity`,
`input_image_identity`, `method`, `repository`, `commit` (40 hex), `code_license`, `checkpoint{name, source,
size_bytes, sha256, weight_license, loaded_evidence}`, `input_size`, `output_size`, `preprocessing`,
`postprocessing`, `threshold`, `alignment`, `device{type, name}`, `interpreter`, `packages`, `seed`,
`run_timestamp`, `inference_time_seconds`, `status`; plus `fold_attribution` for Method A. Any missing field
refuses the result with that field named. `loaded_evidence` must be **positive** evidence that the
configured checkpoint was loaded — an identity check (the loaded tensors' keys and their checksum against
the configured file's), not an inference from "the code ran without error".

**Rationale**: FR-018a/018b exist because of a real upstream behaviour: Method C has a hardcoded
working-directory-relative checkpoint path, catches the missing-file error, prints a warning and continues
with a **randomly initialised decoder** — correct shape, correct value range, so every output-domain check
passes while the numbers are meaningless. Plausible masks are therefore not evidence, and a check that only
looks at the output cannot distinguish the two cases. Method B's checkpoint was verified to contain all
three component keys, which is the kind of identity check this field carries. Two further fields are
separately recorded for the same reason: `code_license` and `weight_license` are distinct (a repository's
licence does not necessarily cover its published weights), and `packages` must be the real versions of the
environment that ran, not the contents of a `requirements.txt`.

**Alternatives considered**:
- *Accept "the model produced plausible output" as load evidence* — rejected: that is exactly the failure
  FR-018a exists to catch.
- *Verify the checkpoint hash here instead of trusting the runner's claim* — not possible for the loaded
  state, which exists only in the runner's environment; but the recorded `sha256` is checked against the
  checkpoint identity the availability report pinned, so a substituted checkpoint is caught even though the
  load itself cannot be re-performed here.
- *One `license` field* — rejected by FR-056 and ONBOARDING §10.

## R7 — Availability check: per-environment, storable, and never inferred from one machine

**Decision**: `availability.py` produces a verdict per method **in the environment it is run in**, covering
checkpoint presence (with the artefact named, where to obtain it and its size), third-party code at the
pinned revision, dependencies (naming the version conflict rather than deferring to inference time),
licence, and compute device, plus remediation steps. Verdicts are persisted as JSON so they can be
retrieved later without re-running (FR-022). A method that needs an accelerator or a legacy interpreter this
machine lacks is **re-checked in a suitable environment before being declared unavailable** (FR-021), never
abandoned on one machine's verdict.

**Rationale**: The development machine has no DL runtime at all — no torch, torchvision, fastai, timm,
ultralytics, mmsegmentation, onnxruntime, albumentations, opencv-contrib or scikit-image — so a check run
here reports all three methods unavailable. That verdict is true and correctly scoped to *this* machine,
which is why FR-017 makes the environment part of the verdict rather than a footnote. Method A's stack pins
fastai 1.0.60 / torch 1.4.0 / torchvision 0.5.0 capping at Python 3.7, and its checkpoints are fastai-v1
pickles deserialisable only by that legacy stack; Method C needs timm and an EfficientNetV2-M encoder;
Method B needs torch plus two geometry dependencies that its own manifest omits. None of those absences is a
property of the method.

**Alternatives considered**:
- *Install the DL runtimes here to run the availability check* — rejected: FR-052 keeps the runtimes out of
  this project's environment, and the check is not more accurate for being run here. It would be accurate
  about a machine that is not going to run inference.
- *Treat an unavailable verdict as final* — rejected by FR-021.
- *Report a single run-level availability* — rejected: availability is per method, and US2 scenario 3
  requires one available method's results to appear normally while the other two are listed as not run with
  reasons.

## R8 — A result produced against a stale page list

**Decision**: A returned result is admitted only if its `page_list_identity` equals the identity of the
export the project currently recognises as current. A result carrying a different identity is **refused**,
not aligned by page name, with the reason naming both identities (received vs expected). If the project has
legitimately re-exported — the page set changed — the new export carries a new identity and previously
admitted results keep theirs; earlier admitted results are never recomputed or invalidated (FR-059), and
the run stays reportable. Re-running against the new export is cheap by design (ONBOARDING §3: inference is
the cheap part, setup is the expensive part), so a runner re-runs rather than having their old result
retrofitted.

**Rationale**: Page identifiers alone would let a result be aligned by name, which silently accepts a
result computed over a page set that no longer exists — the identity would match on every page that happens
to still be present and silently omit or mis-attribute the rest. FR-054 requires the identity to be recorded
with each returned result precisely so this case is detectable. FR-059 requires that admitting a later result
never recomputes an earlier one, so the stale result cannot be "fixed" by rescoring.

**Alternatives considered**:
- *Align by page name and admit with a warning* — rejected: it is a silent partial acceptance, which is the
  class of failure FR-058 exists to eliminate.
- *Invalidate the whole run on a re-export* — rejected by FR-059.
- *Version the page list and accept any older version* — rejected: the benchmark's premise is that all
  methods are scored on the same pages (US3 scenario 3, FR-053(a)); accepting a mix of versions would make
  the comparison table meaningless.

## R9 — Deviations from the pinned revision, and where workarounds live

**Decision**: The three verified upstream defects (a removed numeric-library alias, missing geometry/
graphical imports in a headless environment, and a configuration value the code ignores in favour of a
hardcoded constant) are worked around **in the runner's own environment** and recorded in the provenance
record as a deviation from the pinned revision, together with what was changed and why. Nothing is patched
in-tree, and no patch series, partial copy or single-file extract is committed (FR-055, SC-013). A workaround
that **changes the model's output** rather than merely letting it run — a different threshold, a different
preprocessing constant, a re-exported checkpoint — makes the result a **different method**, not the pinned
one, and it is recorded as such rather than as the pinned method with a caveat.

**Rationale**: The licence positions decide this, and they decide it about repair rather than integration.
Method A is MIT, Method B is GPL-3.0, Method C has **no licence at all** — default copyright, the most
restrictive position of the three despite looking the most modern. Under the requester's 2026-09-10 decision
all three are used as external repositories cloned in the runner's own environment, so the project carries
no derivative work: Method B's copyleft is moot for distribution because there is nothing to distribute, and
Method C's absence of a licence permits exactly the one thing being done with it. But the same absence means
the project cannot lawfully patch C — or any of them — in-tree, which is why FR-055 is absolute about form
and why the deviation record is the only lawful route for a workaround.

**Alternatives considered**:
- *Vendor a patched copy of the two-line fixes* — forbidden by FR-055 and would be the worst licence
  position available (a derivative work of an unlicensed repository, committed).
- *Pin a community fork instead of upstream* — rejected for Method B: the `kha-white` and `Ajatt-Tools` forks
  are patch sources for reference only; the pinned artefact is upstream at an exact revision, so the
  provenance names something stable.
- *Silently apply a workaround and report the pinned revision* — rejected: it would make the recorded
  revision a false statement, and FR-056 refuses incomplete provenance rather than accepting gaps.

## R10 — Where the export and the admitted masks live, given `outputs/` is gitignored

**Decision**: Two committed destinations are added at the repository root. `benchmark/` holds the
distributable export's metadata: `page-list.json` (small, text) and `image-identity.json` (390 hashes). Its
`dist/` subdirectory holds the ~138 MiB image bundle and is gitignored, matching the existing `*.zip` rule.
`deliverables/<run_id>/<method>/` holds admitted results — masks, per-page metadata, metrics, visualisations,
error report — and **is committed**, following ONBOARDING §9's layout.

**Rationale**: These two are the only artefacts in this feature that cannot be regenerated from the
repository, and they cannot be regenerated for different reasons. The export is the shared input three
external runners work from; it is produced once and must be stable. A DL prediction mask requires a
third-party repository at a pinned revision, a ~79 MB–216 MB checkpoint, and an accelerator this machine
does not have — so unlike the classical baseline's masks, which `sweep` can reproduce in seconds on CPU,
the DL masks exist only as the runners' returned bytes. Spec 1's plan line 158 records `outputs/` as
"gitignored; runtime artifacts only (never committed)", which is correct for the classical path and is left
untouched; it simply cannot be the destination for either of these. ONBOARDING §9 is explicit that a mask
delivered as a Drive link does not count, and that the ban in `notebooks/README.md` on committing "toàn bộ
output ảnh lớn" refers to intermediate and debug images, not to prediction masks. The measured mask size
makes this cheap: 390 PNG masks at 1654×1170 binary compress well, and the GT side is 10.4 MiB for the
whole dataset.

**Alternatives considered**:
- *Put everything under `outputs/` and un-ignore it* — rejected: it would un-ignore the classical sweep's
  scratch output as well, and `outputs/` is regenerable by definition while these two are not. Two named
  committed destinations say what they are for.
- *Commit the image bundle* — rejected: ~138 MiB of binaries in history for bytes that the identity record
  already makes verifiable wherever they are obtained.
- *Store admitted masks outside the repository* — rejected by ONBOARDING §9.
- *Commit only metrics and not masks* — rejected: FR-043 requires masks stored so Spec 3 can consume them
  from page id plus method name alone, at full aligned resolution in the FR-004 convention, which means the
  masks themselves.

## R11 — Extending the CLI and the configuration without disturbing Spec 1

**Decision**: Three subcommands are added to the existing `argparse` chain in
`src/manga_text_seg/cli.py`, alongside the current `discover|validate|run|sweep|report|visualize|scorecard`:
`export` (write the distributable page list + image identity + bundle), `admit` (validate a returned
hand-off and admit it into a named run), and `status` (report admitted versus awaited methods with reasons).
`configs/dl.json` is a **separate** configuration file from `configs/default.json`, carrying per method the
external repository path, the exact revision, the checkpoint path and expected sha256, the device, and the
method-level preprocessing/threshold/postprocessing settings. `config.py` gains a loader for it that reuses
the existing `_require_mapping` / `_resolve` helpers and validation style.

**Rationale**: FR-049 requires exactly these three operations, and the existing CLI already establishes the
conventions they need: `_common_parser` gives `-c/--config`, `_output_dir` resolves the run directory,
`ConfigError` exits 2 and `FileNotFoundError` exits 2, and dispatch is an if/elif chain. Reusing those means
the new subcommands are a handful of lines each rather than a second entry point. A separate config file is
deliberate rather than tidy-minded: `configs/default.json` is the frozen classical configuration whose
`methods` list drives `sweep`, and adding DL entries to it would change what `sweep` iterates over —
breaking the Spec 1 baseline's reproducibility, which FR-002 and the constitution's reuse principle both
protect. The DL configuration is also structurally different (repository paths, revisions, checkpoint
hashes, devices) and belongs beside the classical one, not inside it. `config.py`'s frozen `Config` dataclass
and its `{name, params}` method shape are left untouched.

**Alternatives considered**:
- *Add DL methods to `configs/default.json`'s `methods` list* — rejected: it would silently change the
  classical sweep's scope and make the frozen baseline non-reproducible.
- *A second CLI entry point* — rejected: FR-049 asks for operations on this project's CLI, and a second
  entry point would duplicate config resolution and error handling.
- *Subcommand per method* — rejected: methods are data (a registry entry plus configuration), not commands;
  `run --method <name>` and `admit --method <name>` already parameterise by name.
- *A new `Config` class for DL* — rejected under Ponytail: a small loader returning a frozen dataclass in the
  same style is the shortest thing that works, and the shared validation helpers already exist.

## R12 — Stand-in testing: proving both paths meet at one evaluation procedure

**Decision**: Default tests never load a real checkpoint or download a model (FR-051). The adapter contract
is exercised by a stand-in adapter that returns a synthetic binary mask for a two-page fixture, asserted
against the same assertions a real adapter must satisfy: a binary mask of aligned size, the four metrics, an
inference time, and a metadata record. A second test expresses the *same* synthetic result as a hand-off
produced "elsewhere" — written as files plus provenance — admits it through `receipt.py`, and asserts the
metrics are **byte-identical** to the adapter path's. Receipt tests cover the refusal cases directly: a
stale `page_list_identity`, a mismatched `input_image_identity`, a missing FR-018b evidence field, a
wrong-sized mask, and an out-of-convention value set, each refused with the specific reason named and
nothing written into the run. Integration tests against real checkpoints are separated and opt-in.

**Rationale**: US1's independent test is exactly this two-path equality, and it is the strongest available
evidence for FR-053(e) — that one evaluation procedure scores both a locally-driven adapter and a
remotely-produced hand-off — without needing a GPU or a checkpoint in CI. It also discharges US1 scenario 6
directly: because the stand-in is registered through the same registry and scored by the unmodified
`metrics.compute_metrics`, the test fails if anyone adds method-specific branching to the metric or
reporting modules. Asserting "byte-identical" rather than "approximately equal" is possible because the
metrics are integer-count ratios over the same bytes, and it is the wording the acceptance scenario uses.

**Alternatives considered**:
- *Test against a small real model* — rejected by FR-051 and by the constitution's no-speculative-tests
  principle: it would need a download and a runtime this project deliberately does not have.
- *Test the receipt path with a mocked metrics function* — rejected: the point of the test is that the real
  evaluation procedure scores both paths identically; mocking it would test the mock.
- *Compare metrics with a tolerance* — rejected: the paths produce the same integers from the same bytes;
  a tolerance would hide a real divergence.

## R13 — Reporting across methods: what is compared and what is not

**Decision**: One combined run (`benchmark.py`) produces one summary table and one set of charts with
IoU/Precision/Recall/F1 plus processing time per method, mean and standard deviation, over the same page
list. Timing is reported per method **labelled with its producing device** and is never ranked across
devices (FR-060) — a time chart or table may exist, but no cross-device ordering is presented. A method not
yet returned appears as "not run" with its awaited status and reason, and **no placeholder or imputed
numeric value anywhere** (FR-036). Pixel Accuracy is not a primary ranking metric and no OCR, translation or
inpainting measure ranks any method. The contamination disclosure (SC-011) appears next to the scores: A
de-contaminated by LOFO, C fully contaminated, B's extent unknowable. Per-method failure lists distinguish
"pages that failed inside a method that ran" from "method did not run", and an absent method is never
presented as having zero failures.

**Rationale**: Accuracy is comparable because FR-053 fixes the six shared invariants — same page list, same
input bytes, same alignment, same mask format, same evaluation procedure, recorded environment metadata —
and a deviation in any one means the result is not admitted. Timing is not comparable for the opposite
reason: three runners run on three different machines, so a time difference measures the hardware as much as
the method. FR-060 requires the device to be named specifically — `"cuda"` alone is not enough, the GPU name
is required — precisely so a reader can see that the times are not a ranking. The "no placeholder value"
rule matters more than it looks: a zero or a blank in a comparison table reads as a score, and a zero for a
method that never ran would rank it last rather than mark it absent.

**Alternatives considered**:
- *Rank methods by time* — forbidden by FR-060 and meaningless across three machines.
- *Emit `null` metrics for awaited methods so the table is rectangular* — rejected: it invites aggregation
  that treats absent as zero. The row is present with an explicit not-run status.
- *Report Pixel Accuracy alongside* — rejected: on a background-dominated page it is dominated by the
  background, and US3 scenario 7 excludes it as a primary ranking metric.
- *Defer the whole table until all three return* — rejected by FR-059: the run must remain reportable while
  methods are outstanding.
