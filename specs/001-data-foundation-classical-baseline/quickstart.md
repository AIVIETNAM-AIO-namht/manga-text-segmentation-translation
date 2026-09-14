# Quickstart: Data Foundation & Classical Baseline

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Config**: single JSON file (FR-036)

## 0. Prerequisites
- Python 3.11+, CPU only. Repo root `d:\UTE\DIPR\Project`.
- Verify environment (versions recorded, not assumed):
  `python --version` and `pip list` (expect opencv-python, numpy, pandas, matplotlib, pytest).
- Dataset present and read-only: `data/raw/...`, `data/groundtruth/post-processed/...`.
  Never write under `data/`; never open `data/no-need-to-read/`.

## 1. Install (dev)
```bash
pip install -e ".[dev]"   # package: manga-text-seg  (name fixed at task time)
pytest --collect-only -q  # sanity: suite collects
```

## 2. Discover + validate (US1)
```bash
manga-text-seg discover  --config configs/default.json            # -> manifest.json (390 pairs)
manga-text-seg validate  --config configs/default.json            # -> validation-report.json (60 orphans, 0 corrupt)
diff <(manga-text-seg discover --config configs/default.json) manifest.json  # byte-identical re-run
```

Expected: 450 masks discovered → 390 paired, 60 orphans across the 6 imageless manga,
`EvaLady/006` paired with `all-white-empty` encoding.

## 3. Run one method, then the sweep (US2–US3)
```bash
manga-text-seg run   --config configs/default.json --method otsu   # single method, full manifest
manga-text-seg sweep --config configs/default.json                 # all configured methods
```
Outputs per method: `outputs/segmentation/<run_id>/<method>/<manga>/<NNN>.png + .json`
(single-channel uint8 {0,255}, 1170×1654). One slow/failing image never aborts the run —
check `failures.json`.

## 4. Report + visualize (US4–US5)
```bash
manga-text-seg report    --config configs/default.json   # metrics.csv + summaries.json (IoU/P/R/F1 — never Pixel Accuracy)
manga-text-seg visualize --config configs/default.json   # viz/<method>/first-20/... (default first-N; best-N/worst-N by F1 via config)
```

## 5. Add a new classical method (SC-008 drill)
1. New file `src/manga_text_seg/methods/my_method.py` implementing `name` + `segment(image)`.
2. One `register("my_method", factory)` line.
3. Contract test asserting {0,255} output; acceptance test on the smoke subset.
4. List it in config `methods:` and re-run `sweep` — metrics/output format unchanged.
