# Contracts

Three new contracts. Spec 1's four contracts and Spec 2's three are **reused, not restated** —
[Spec 1's manifest and mask contracts](../001-data-foundation-classical-baseline/contracts/index.md)
supply the page list, the mask convention and the aligned space;
[Spec 2's contracts](../002-deep-learning-segmentation-benchmark/contracts/index.md) supply the
prediction masks and their sidecars;
[Spec 3's inpainted-output contract](../003-text-removal-inpainting/contracts/inpainted-output.md)
supplies the inpainted page this feature renders onto.

| Contract | File | Covers (FRs) |
|----------|------|--------------|
| Rendered output | [rendered-output.md](rendered-output.md) | FR-046, FR-048, FR-049, FR-052 |
| Pipeline metadata schema | [pipeline-metadata.schema.json](pipeline-metadata.schema.json) | FR-047, FR-035 |
| Error report | [error-report.md](error-report.md) | FR-051, FR-055 |
