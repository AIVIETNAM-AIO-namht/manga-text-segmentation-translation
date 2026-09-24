# Contracts: Text Removal & Image Inpainting

**Date**: 2026-09-24 | **Spec**: [spec.md](../spec.md)

Three new contracts. `inpainted-output.md` is normative for the output tree, its addressing rule and
its artifacts — it is the contract **Spec 4 and Spec 5 read**. `sample-metadata.schema.json` is
normative for the per-sample JSON record. `intake-validation.md` is normative for the procedure this
project applies to a hand-off it receives.

Spec 1's four contracts remain in force and are **reused, not restated**:
[method-interface.md](../../001-data-foundation-classical-baseline/contracts/method-interface.md)
(the `segment(image) -> mask` interface — this feature consumes its output, never implements it),
[manifest.schema.json](../../001-data-foundation-classical-baseline/contracts/manifest.schema.json)
(the internal manifest, FR-001), [metrics.schema.json](../../001-data-foundation-classical-baseline/contracts/metrics.schema.json)
(the metric record FR-034's selection rules read) and
[output-layout.md](../../001-data-foundation-classical-baseline/contracts/output-layout.md)
(the mask sidecar convention and the aligned mask size this feature's intake re-verifies).

Spec 2's three contracts are likewise **reused, not restated**:
[page-list.schema.json](../../002-deep-learning-segmentation-benchmark/contracts/page-list.schema.json)
(the page list whose `page_list_identity` and `aligned_size` this feature quotes verbatim),
[returned-result.md](../../002-deep-learning-segmentation-benchmark/contracts/returned-result.md)
(the admitted hand-off layout this feature reads, and the receipt validation whose *result* it depends
on without re-running it) and
[provenance.schema.json](../../002-deep-learning-segmentation-benchmark/contracts/provenance.schema.json).

| Contract | File | Covers (FRs) |
|---|---|---|
| Inpainted output tree + addressing | [inpainted-output.md](inpainted-output.md) | FR-025, FR-026, FR-027, FR-028, FR-029 |
| Per-sample metadata record | [sample-metadata.schema.json](sample-metadata.schema.json) | FR-018, FR-022, FR-025 |
| Intake validation + rejection | [intake-validation.md](intake-validation.md) | FR-008–FR-015 |
