# Contract: Inpainted Output Tree + Addressing

**Normative for**: FR-025, FR-026, FR-027, FR-028, FR-029.

This is the contract **Spec 4 and Spec 5 read**. Spec 4's FR-012 selects an inpainting source by a
Spec 3 run ID and its algorithm; Spec 5's Assumptions define the "processed mask" as the post-dilation
mask exported alongside the raw prediction mask copy. Both readings are fixed here.

## Layout

```text
outputs/inpainting/<run_id>/
├── run.json                       # InpaintRun (data-model.md)
├── performance.json               # FR-022 aggregate, by method × algorithm
├── errors.json                    # FR-012 nine categories, every failure
├── selection.json                 # FR-034 rule → selected image_ids, per method
├── qualitative.md                 # FR-035 scaffold — ratings and observations empty
├── qualitative.json               # same scaffold, machine-readable, same emptiness
├── boards/<method>/<rule>/<manga>_<stem>.png   # FR-033 five-panel boards
└── <method>/<algorithm>/<manga>/<page_id>.png  # the inpainted page
                          <manga>/<page_id>.json # SampleMetadata sidecar
                          <manga>/<page_id>.raw.png       # raw prediction mask copy
                          <manga>/<page_id>.mask.png      # post-dilation mask (processed)
                          <manga>/<page_id>.page.png      # the original page image

outputs/inpainting/ablation/<run_id>/   # FR-024, same shape, excluded from the main comparison
```

`<method>` is one of exactly four: `classical_baseline`, `manga_text_segmentation`,
`comic_text_detector`, `unetpp_efficientnetv2`. `<algorithm>` is exactly two: `telea`, `ns` — both
always produced for every page in the main benchmark (FR-019). `<manga>/<page_id>` is the page's
`image_id` split on its separator, so the path spells the `image_id` back out.

## Addressing (FR-029)

A consumer knowing only **(run ID, segmentation method, `image_id`)** locates the page's inpainted
image and metadata at:

```text
outputs/inpainting/<run_id>/<method>/<algorithm>/<manga>/<page_id>.png
outputs/inpainting/<run_id>/<method>/<algorithm>/<manga>/<page_id>.json
```

No index file, no manifest lookup and no run-specific knowledge is required. This is a deliberate
choice: a run-level index would satisfy "locate" only for a consumer willing to read the index, which
is run-specific knowledge, and it would make a partially-copied run directory unusable (research R7).

## Artifact rules

1. The inpainted page is written at exactly the aligned page size (1654×1170), in the configured
   output format, default PNG and lossless (FR-028, SC-002).
2. `<page_id>.raw.png` is a **byte-identical copy** of the prediction mask as admitted under
   `deliverables/<run_id>/<method>/masks/`. It is the same object, not a re-derivation of it, so
   "Spec 3's raw copy" and "Spec 2's output" cannot drift apart.
3. `<page_id>.mask.png` is the mask **actually handed to `cv2.inpaint`** — after normalization,
   alignment and dilation. This is Spec 5's "processed mask". It is written as its own artifact so a
   board can label which mask it is showing without re-deriving it.
4. `<page_id>.json` is a `SampleMetadata` record satisfying
   [sample-metadata.schema.json](sample-metadata.schema.json).
5. Ground truth appears in no artifact of this tree. No board panel is a GT mask or a
   prediction-vs-GT overlay (FR-033).
6. A run never overwrites a previous run's outputs. A repeated run ID refuses unless overwrite was
   explicitly requested (FR-027, SC-006). This is what makes the gitignored `outputs/inpainting/`
   safe to leave untracked (research R6).

## What this contract does not do

It does not re-evaluate segmentation. Nothing in this tree feeds Spec 1's or Spec 2's metrics
(FR-030); the raw mask copy exists for provenance and for the qualitative boards, not as a new
segmentation result.
