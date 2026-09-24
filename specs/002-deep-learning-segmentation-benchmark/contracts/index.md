# Contracts: Deep Learning Segmentation Benchmark

**Date**: 2026-09-19 | **Spec**: [spec.md](../spec.md)

Three new contracts. Schemas are normative for file artifacts; `returned-result.md` is normative for
the receipt-validation procedure the project applies to a hand-off.

Spec 1's four contracts remain in force and are **reused, not restated**:
[method-interface.md](../../001-data-foundation-classical-baseline/contracts/method-interface.md)
(the `segment(image) -> mask` interface every adapter satisfies, FR-008),
[manifest.schema.json](../../001-data-foundation-classical-baseline/contracts/manifest.schema.json)
(the internal manifest, FR-001 — note it is **not** the distributable page list),
[metrics.schema.json](../../001-data-foundation-classical-baseline/contracts/metrics.schema.json)
(the metric record shape FR-057 produces) and
[output-layout.md](../../001-data-foundation-classical-baseline/contracts/output-layout.md)
(the mask sidecar convention the DL path extends with its own richer sidecar).

| Contract | File | Covers (FRs) |
|---|---|---|
| Distributable page list + image identity | [page-list.schema.json](page-list.schema.json) | FR-054, FR-054a, FR-001/FR-002 boundary |
| Returned result + receipt validation | [returned-result.md](returned-result.md) | FR-053, FR-057, FR-058, FR-059, FR-050a |
| Provenance record | [provenance.schema.json](provenance.schema.json) | FR-056, FR-018a, FR-018b, FR-044, FR-060 |
