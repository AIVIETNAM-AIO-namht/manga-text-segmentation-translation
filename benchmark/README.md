# `benchmark/` — the distributable export

What leaves this project and goes to the runners. Produced by:

```bash
manga-text-seg export --config configs/default.json
```

Derived from Spec 1's internal manifest (`outputs/segmentation/default/manifest.json`), which it is
**not**: the manifest carries `gt_root`, `raw_root` and per-page `mask_path`. None of those may appear
here (FR-054a — *the images are the question and the masks are the answer*).

## Contents

| Path | Committed | What it is |
|---|---|---|
| `page-list.json` | yes | The page list: 390 pages sorted `(manga, stem)`, each with `image_id`, `image_ref` and `input_image_identity`, plus `page_list_identity`, `aligned_size`, `alignment` and `mask_convention`. Schema: [`contracts/page-list.schema.json`](../specs/002-deep-learning-segmentation-benchmark/contracts/page-list.schema.json) |
| `image-identity.json` | yes | The 390 sha256 digests of the raw image bytes. Small, and the only thing that makes a copy of the bundle verifiable |
| `dist/images/` | **no** — gitignored | The ~138 MiB image bundle. Distribution, not contract; the identity record above is the contract |

## Rules

- **Generated, never hand-edited.** Re-running `export` yields a byte-identical `page-list.json` and the
  same `page_list_identity`. An edit breaks that, and every sidecar quoting the old identity is refused
  on receipt (check 1).
- **A runner quotes `page_list_identity` and each `input_image_identity` verbatim.** A runner MUST NOT
  recompute either by their own method.
- **No ground truth, in any form.** No `gt_root`, no `mask_path`, no GT pixels.
- **How the bundle travels does not matter.** Drive, a shared disk, a USB stick — `image-identity.json`
  is what verifies it on arrival.

Every runner consumes the same list instance (FR-054).
