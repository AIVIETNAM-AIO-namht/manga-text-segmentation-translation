# Specification Quality Checklist: Deep Learning Segmentation Benchmark

**Purpose**: Validate specification completeness and quality before proceeding to planning
**Created**: 2026-09-10
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

## Requirement Completeness

- [x] No [NEEDS CLARIFICATION] markers remain (count: **0**; three were raised and all three were
      answered on 2026-09-10 — see Notes)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Success criteria are technology-agnostic (no implementation details)
- [x] All acceptance scenarios are defined
- [x] Edge cases are identified
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

## Feature Readiness

- [x] All functional requirements have clear acceptance criteria
- [x] User scenarios cover primary flows
- [x] Feature meets measurable outcomes defined in Success Criteria
- [x] No implementation details leak into specification

## Notes

### Items intentionally left unresolved

**None.** `[NEEDS CLARIFICATION]` count: **0**.

Three clarification questions were raised against this specification. All three were put to the
requester in a single decision round and all three were answered on **2026-09-10**. Each answer is
recorded below and bound into `spec.md` as a requirement rather than left as prose.

### Clarification questions, answered 2026-09-10

1. **Sequencing against Spec 1.** Spec 1 exists as a specification only — no manifest, loader,
   alignment code or metric module is present — so the ordering genuinely changed this feature's scope
   and could not be answered from the repository.
   - **Answer.** Spec 2 is finalised against the contracts Spec 1 defines, but need not be implemented
     immediately. Spec 1 is implemented first, then the external model runs, then benchmark integration.
   - **Bound as.** FR-002 rewritten: Spec 1's implementation is a **blocking precondition**, not scope;
     the availability check reports it unmet while absent and MUST NOT substitute a loader, an alignment
     rule or a locally derived page list. Four new *Out of Scope* bullets make "building any part of
     Spec 1" explicitly out of scope. `## Assumptions` → *Dependency on Spec 1* records the confirmed
     sequence.

2. **Method A's legacy-environment requirement.** Method A's stack caps at Python 3.7 while the
   interpreter installed here is 3.10.1 with no virtual environment, and its checkpoints can only be
   deserialised by that legacy stack. Building a dedicated isolated environment versus reducing the
   feature to a two-method comparison differed materially in scope.
   - **Answer.** Each team member runs one model independently in Google Colab; the models need not share
     one local environment. Method A is attempted in its required Colab environment, and if it cannot run
     it is reported unavailable with the reason.
   - **Bound as.** A new requirement group **FR-052…FR-060** (*Distributed execution and result
     hand-off*), which also elevates the requester's five named invariants — shared page list,
     preprocessing/alignment convention, prediction-mask format, evaluation procedure, recorded
     environment metadata — into FR-053 as the admissibility test for a returned result. FR-053 carries
     **six** invariants rather than five: this specification adds the sixth, **the input images the page
     list denotes** (**FR-054a**), because moving execution into three separate Colab sessions removes the
     assumption the requester's list rested on — that the images were already locally available to whoever
     ran a model. FR-054a requires the images to be distributed with a per-page verifying identity and
     every returned result to echo it, and refuses a runner that obtained its images from any other
     source, since accuracy comparability depends on identical bytes and not merely on identical names. It
     is threaded through FR-058 (receipt validation), FR-050a (a rejection test), FR-049 (CLI export), two
     Key Entities and SC-012. FR-017 now scopes an availability verdict to the environment that produced
     it. The *Interpreter-version ceiling* and *No compute device* edge cases were rewritten from scope
     forks into attempt-then-report. FR-050a adds six receipt-path tests. SC-012 verifies it.
     `## Assumptions` → *Execution model* records the decision and the sixth invariant's justification.

3. **Licence position across all three methods.** Verified as three *different* positions: MIT (A),
   GPL-3.0 (B), and no licence stated anywhere (C). Method C's silence meant no grant to copy, modify or
   redistribute, so the integration mode — vendored-and-patched versus configured-external-repository
   versus reimplemented — could not be settled from evidence.
   - **Answer.** Use external repositories. Team members may clone and run the originals in Colab, but the
     main project must not vendor or copy third-party source code or weights. The project stores only
     shared contracts, evaluation code, configurations, standardised outputs and documentation, and
     records repository, commit, checkpoint, licence and environment for every method.
   - **Bound as.** **FR-055** (no vendoring, in any size or form — FR-046 extended to cover patch series
     and single-file extracts) and **FR-056** (provenance record naming repository, exact revision,
     checkpoint identity, code licence, weight licence, device, interpreter and package versions, and run
     time; a result with a missing or incomplete record is refused). SC-013 audits the tree for zero
     third-party source and zero weight files; SC-014 audits provenance completeness at 100%. FR-018b
     requires positive checkpoint-loading evidence in the provenance record, because under FR-052 the
     project never witnesses the run. A new *Upstream defect requiring a workaround* edge case requires
     workarounds to live in the runner's environment and declares an output-altering workaround a
     different method. `## Assumptions` → *Repository boundary* and *Licence* record both.

### Questions deliberately resolved rather than asked

One topic was a candidate for clarification and was resolved from evidence instead, because a
requirement or an inherited contract already answered it:

- **Training-data contamination and the possible leave-one-fold-out view.** The contamination is now
  *proven* rather than suspected (see evidence table), and the user forbids new benchmark subsets, so the
  primary evaluation stays the full manifest with mandatory disclosure (SC-011). A supplementary
  de-contaminated reporting view is recorded as available but explicitly **not** required, because the
  constitution's simplicity principle forbids building a reporting artefact nobody asked for. Asking
  would have invited scope growth that the conservative reading of the user's constraint already
  settles. **Superseded in part on 2026-09-10 (clarification round 2, question 2):** the requester chose
  to make leave-one-fold-out method A's *primary* checkpoint strategy (FR-013a) — the full-manifest rule
  and the disclosure requirement stand, but the "not required" position no longer holds for method A.
  Method C, which publishes a single checkpoint, remains as originally recorded.

One topic previously listed here was **superseded by answer 2** and is recorded as such rather than
removed:

- **Compute device — formerly "resolved by inheritance".** This checklist previously recorded that the
  compute device was not asked because Spec 1 FR-037 mandates CPU operation and this feature inherits
  that constraint. Answer 2 makes that inheritance *partial* rather than total: the CPU mandate binds
  what the project itself executes — the evaluation procedure, metrics, reporting, visualisation and the
  classical baseline — which all remain CPU-only and reproducible without an accelerator. It cannot bind
  per-method inference, which runs in hosted notebook environments chosen precisely because they supply
  an accelerator. **FR-060** now binds the consequence: every timing figure carries its producing device,
  and no cross-method timing ranking is presented where devices differ. Accuracy is unaffected and stays
  directly comparable. `## Assumptions` → *Compute device and runtime budget* records the re-scoping.
  This is a real reduction in what the timing comparison can claim against the user's original "so sánh
  thời gian xử lý" goal, and it is stated as such rather than presented as an unqualified success
  criterion.

### Deliberate non-binding choices (correct at specification stage)

- **All three methods are verified; none is left unspecified.** An earlier revision left methods A and C
  unnamed because they had not been verified. Both have since been verified against primary sources —
  repository, licence, architecture, checkpoint identity and size, reachability, preprocessing,
  threshold, output domain and entry point — and are recorded in `## Assumptions`. This satisfies the
  user's instruction "Không tự giả định checkpoint hoặc API nếu chưa kiểm tra" in the positive direction:
  nothing is assumed, and nothing is named without evidence.
- **Technology stack remains non-binding.** Following Spec 1's convention, the stack is recorded in
  `## Assumptions` and fixed at planning time. The deep-learning runtime is flagged as an unavoidable
  new dependency requiring justification against the constitution's simplicity-and-reuse principle —
  this feature cannot be delivered without it, which is the test that principle sets. Under FR-055 that
  dependency lands in the runners' environments, not in the project's, which materially narrows what
  the project itself must add.
- **Metric convention for degenerate pages is required but not chosen here.** FR-032 mandates that a
  convention be declared, recorded and applied identically to every method; which convention is an
  implementation decision that must match whatever Spec 1's metric module already does, and cannot be
  set here without reading code that does not yet exist.
- **The page-list and image-distribution formats are required but not chosen here.** FR-054 requires the
  page-list export to be self-contained, identity-carrying and consumable without access to the project's
  data tree, and requires it to carry no ground-truth reference and no ground-truth pixels. FR-054a
  requires the input-image distribution to carry a per-page verifying identity strong enough for the
  project to confirm on receipt which bytes a runner actually processed. Neither serialisation, nor the
  strength or form of that identity, is fixed here: both are implementation decisions for planning, and
  both must be consistent with whatever manifest schema Spec 1 ships — which does not exist yet.

### Evidence base

Every quantitative claim in `spec.md` was measured directly against the dataset or read from primary
source, not inherited from a subagent summary. Where a subagent report was the lead, the decisive
numbers were re-derived locally before being written into the specification:

| Claim | How verified |
| --- | --- |
| 450 masks / 390 pairs / 60 orphans | Filesystem enumeration of both trees |
| All 390 pairs share one size combination (1654×1170 raw, 1656×1176 mask) | Read of all 390 image and mask headers |
| The +2/+6 delta is exactly pad-to-multiple-of-8 | All 450 masks measured as multiples of 8; `ceil(1654/8)×8 = 1656`, `ceil(1170/8)×8 = 1176` |
| **The evaluation ground truth IS the published training dataset of methods A and C** | Upstream `trainFolders` (45 books) fetched from source and compared set-wise against the local ground-truth tree: **identical, zero difference either way**; upstream dataset record states 450 images, post-processed = small components removed, holes filled, dimensions padded to multiples of 8 — each clause matched against a local measurement |
| Contamination is 5-fold cross-validated over those 45 books, one checkpoint per fold | Read of the upstream fold construction in source (KFold, shuffle, fixed seed 42) |
| Ground truth is stroke-level, not region-level | Area fraction 0.86–11.9%, 69–724 components/page, area:bbox 0.42–0.65 |
| One-pixel boundary shifts dominate the metric | Stroke width 3.06–14.50 px; IoU vs. own 1× erode as low as 0.446 |
| Ground truth is not artificially dilated | IoU(mask, dilate) > IoU(mask, erode) on all 6 sampled pages |
| Method A repository, licence, architecture, checkpoint set, Python ceiling | Source read of the original repository; release tag inspected; candidate identically-named repository confirmed non-existent (HTTP 404) |
| Method A pads input to a multiple of 8, fixed 0.5 threshold, multi-class label space | Source read of the training data pipeline and evaluation code |
| Method C repository, absence of any licence, architecture, output domain | Source read; licence absence checked on both the repository and the model card |
| Method C checkpoint size and reachability | HTTP inspection of the hosted artefact (216,911,417 bytes) |
| Method C silently continues with random weights when the checkpoint is missing | Source read of the checkpoint-loading error handler |
| Method C pads to a multiple of 32, not 8; aligned size is a multiple of 8 but not 32 | Source read, then arithmetic against the measured page size |
| Method B architecture, preprocessing, output resolution | Source read; committed example output at exactly 1654×1170 |
| Method B checkpoint reachability and contents | HTTP HEAD 200; downloaded and key-inspected (79,948,869 bytes) |
| Method B numpy-2 incompatibility | Alias removal confirmed against installed numpy 2.2.6 |
| Method B training-data contamination extent (≈1/3, split unpublished) | Upstream README's own statement |
| No Spec 1 implementation exists | Search for `src/`, `tests/`, `outputs/`, manifest, config, CLI: all absent |
| No deep-learning runtime installed | Package inventory of the active Python 3.10.1 environment |
| No checkpoints and no vendored third-party model code anywhere under the project root | Filesystem search for weight files and for the three method repositories: all absent |
| Three upstream defects block a clean headless run (removed numeric-library alias; graphical-component import; ignored threshold parameter) | Source read of each, per method B and method C, and confirmed against the installed numeric-library version |

Two subagent claims were **refuted** and excluded rather than carried forward:

1. That Manga109s annotations are region-level rectangles while model outputs are stroke-level. The
   measurements above show the annotations are themselves stroke-level, so this cannot be used to
   explain low IoU. Recorded in `## Assumptions` as a false claim.
2. That method C's repository is a fork of method A's, and that an identically-named alternative
   repository for method A was a viable candidate. The former is an independent reimplementation with no
   stated licence; the latter does not exist (HTTP 404). Both corrections are reflected in the
   specification.

Two earlier specification assumptions were **superseded by session evidence** and rewritten: the
*Methods A and C — deliberately unspecified* assumption (both are now specified) and the *Method B
training-data contamination* assumption (contamination is benchmark-wide and proven, not method-B-only
and unknowable).

One earlier **checklist** position was superseded by the requester's answer 2 and is retained above with
its supersession stated: the "compute device resolved by inheritance" note.

### Validation history

- Iteration 1: content quality, requirement completeness and feature readiness checked against
  `spec.md` as written. All items pass with the single intentional `[NEEDS CLARIFICATION]`.
- Iteration 2: re-validated after the Method A and Method C research and the ground-truth provenance
  discovery. Added FR-018a, FR-024a, FR-024b, three edge cases and a revised SC-011 to bind
  requirements to the newly verified facts; rewrote four assumptions; resolved two candidate
  clarification questions from evidence instead of asking them. All checklist items still pass; the
  `[NEEDS CLARIFICATION]` count remains 1.
- Iteration 3 (this session): re-validated after the requester answered all three clarification
  questions on 2026-09-10. Added the FR-052…FR-060 distributed-execution group and FR-018b and FR-050a;
  amended 15 existing FRs (FR-001, FR-002, FR-008, FR-012, FR-016, FR-017, FR-033, FR-034, FR-037,
  FR-038, FR-042, FR-044…FR-047, FR-049) to work across a trust boundary the project never witnesses;
  added three Key Entities (Distributable Page List, Returned Result, Provenance Record) and revised
  three more; added four Out of Scope bullets; amended SC-001, SC-003, SC-007, SC-008 and added
  SC-012…SC-014; added six edge cases and rewrote three; rewrote User Stories 1–4 for the distributed
  model; rewrote or added seven `## Assumptions` subsections. Self-caught and fixed one contradiction
  between the Distributable Page List entity and the *Ground truth reaching a runner* edge case (the
  entity carried a ground-truth reference the edge case forbids).
- Iteration 4 (this session): re-validated the distributed-execution group against the Key Entities and
  found one real gap. FR-054 requires the page list to be consumable *without access to the project's own
  data tree*, but no requirement supplied the **raw image bytes** a runner needs in order to process a
  page. Three runners could each obtain their images from a different source — a different release of the
  same dataset, a re-download, a re-compressed copy — and the page names and even the page sizes would
  still match, so no check already in the specification would catch it while the accuracy comparison
  silently stopped meaning anything. Closed by adding **FR-054a** (distribute the images with a per-page
  verifying identity; refuse any runner that sourced them elsewhere), widening **FR-053** from five
  invariants to six, adding a seventh edge case (*Runner sourced its images elsewhere*), and threading
  FR-054a through FR-058, FR-050a, FR-049, SC-012, both affected Key Entities, the *Execution model*
  assumption and the Status line. The sixth invariant is recorded as **this
  specification's addition, not the requester's** — the requester's five are reproduced faithfully, and
  the sixth exists because the move to distributed execution removed the assumption those five rested on.
- Iteration 5 (this session): re-validated after the requester answered three further clarification
  questions on 2026-09-10, cross-referenced against the course outline (đề cương) in `ref/`. Question 1
  fixed the evaluation scope at all 390 pairs for all four methods (FR-001 amended; the outline's
  ~100–200-image subset suggestion deliberately not adopted). Question 2 chose a full leave-one-fold-out
  checkpoint strategy for method A — adding FR-013a and amending FR-058, SC-011, the *Training-data
  contamination* and *Method A* assumption subsections and the Status preamble — so method A's reported
  score is de-contaminated by construction. Question 3 pinned upstream `dmMaze/comic-text-detector` at a
  pinned revision as method B's primary source, with forks as recorded patch sources only (Method B
  assumption subsection amended). Final counts: **67 functional requirements, 14 success criteria,
  24 edge cases, 0 `[NEEDS CLARIFICATION]` markers**. All checklist items pass; no checkbox state
  changed (16/16 → 16/16).
  Final counts: **66 functional requirements, 14 success criteria, 24 edge cases, 0
  `[NEEDS CLARIFICATION]` markers**. All checklist items pass.

