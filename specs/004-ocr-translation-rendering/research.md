# Research: OCR, Translation & Text Rendering

**Date**: 2026-09-24 | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

Ten decisions, each in Decision / Rationale / Alternatives considered form. Every unknown raised in
the plan's Technical Context is resolved here; none is left as NEEDS CLARIFICATION.

---

## R1 — Reuse Spec 1's mask, alignment and image-IO helpers rather than writing new ones

**Decision**: `intake.py` and `regions.py` call Spec 1's existing helpers directly —
`imaging.load_raw_page`, `imaging.load_mask`, `imaging.save_mask`, `imaging.MASK_ENCODINGS`,
`normalize.normalize_mask`, `align.align`/`AlignedPair`, and `provenance.require_complete` where a
sidecar must be complete before use. No mask loader, no binariser and no aligner is written in this
feature.

**Rationale**: These helpers already encode the conventions FR-002 requires — single channel, uint8,
background 0 / text 255, aligned space 1654×1170, no resizing — and Spec 1's tests already pin them.
Reimplementing them would create a second definition of "binary mask" that could drift from the one
Spec 1, Spec 2 and Spec 3 all use, and FR-002 explicitly says *reuse* the convention. Spec 3's plan
made the same choice toward the same helpers, so all three downstream features read a mask the same
way.

**Alternatives considered**: A local `maskio.py` in this feature (rejected — a second binarisation
policy, and the polarity question Spec 1 already settled would have to be re-litigated);
copying Spec 1's `normalize.py` into this feature (rejected — copy-paste implementation drift, the
exact thing the coding standard forbids).

---

## R2 — How the default segmentation method is resolved, and why Spec 2's `availability.py` is not consulted

**Decision**: FR-010's default is resolved by **reading Spec 2's benchmark selection from a
configured path** — a selection record under the Spec 2 outputs, named in
`configs/translation.json`. Resolution order is: explicit configuration → Spec 2's recorded
selection → **refuse to run** with a clear error. Spec 2's `availability.py` is **not imported** and
its runtime-probing behaviour is not reused.

**Rationale**: FR-010 asks which method Spec 2's *benchmark* selected, which is a result, not a
capability probe. `availability.py` answers a different question — whether a DL runtime is importable
in the current interpreter — and importing it would pull Spec 2's optional-runtime machinery into a
feature that FR-005 forbids from containing any segmentation runtime. Reading a recorded result keeps
the dependency at the file level and keeps the refusal path honest: with no selection and no
configuration the feature refuses rather than picking, which is what FR-010 demands.

**Alternatives considered**: Importing `availability.py` and defaulting to the first available method
(rejected — answers capability, not benchmark outcome, and violates the spirit of FR-005);
defaulting to `classical_baseline` (rejected — arbitrary, and FR-010 explicitly forbids choosing
arbitrarily); reading Spec 1's `metrics.csv` and ranking it here (rejected — that is Spec 2's
selection logic, and re-deriving it here would put two selection policies in the repository).

---

## R3 — `manga-ocr` is an optional extra behind a lazily-imported adapter

**Decision**: `manga-ocr` is declared as an **optional extra** in `pyproject.toml`
(`[project.optional-dependencies] ocr = ["manga-ocr>=0.1.13"]`), never a base dependency. It is
imported **inside** `ocr.py`'s real-engine implementation, at first use, never at module import
time. The adapter protocol and a fixture recognizer live in the same module and import nothing heavy,
so `pipeline.py` and the whole default test suite run with `manga-ocr` absent. A run that selects the
real engine without the extra installed fails with one clear message naming the install command.

**Rationale**: FR-022 names manga-ocr as preferred but requires an adapter so an equivalent tool can
substitute, and FR-028 forbids training it. FR-060 requires the default suite to run *without* a real
OCR engine, and FR-062 requires CPU-only operation. An optional extra satisfies all four at once: the
base install stays light, the fixture suite never touches the dependency, and the engine is swappable
by implementing a protocol rather than by editing the pipeline. The lazy import is what makes the
fixture path real rather than nominal — a module-level `import manga_ocr` would make the dependency
mandatory the moment `ocr.py` is imported, which is exactly the coupling FR-060 forbids. Weights are
downloaded by the library at first use into a gitignored cache and never trained (FR-028, SC-013).

**Alternatives considered**: A base dependency (rejected — every install pays for torch and
transformers to run a fixture test suite); vendoring the model (rejected — SC-013 forbids it
outright, any size, any form); a hand-written OCR (rejected — no requirement asks for one, and
Japanese OCR is not a few lines); calling a cloud OCR API (rejected — FR-022 names a local engine and
the spec's Assumptions record the download-at-setup posture).

---

## R4 — The translation HTTP client is stdlib `urllib.request`

**Decision**: `translate.py` issues its provider request with `urllib.request` from the standard
library — one POST with a JSON body, an `Authorization` header read from the environment, and a
socket timeout. No new dependency is declared for HTTP.

**Rationale**: The requirement is one JSON POST with a timeout and a status code (FR-029, FR-032).
The stdlib does that in a handful of lines, and it is the only HTTP path in the feature. `requests`
and `httpx` are both present in the development environment but **neither is declared** in
`pyproject.toml`; adopting either would mean adding a dependency whose entire contribution is
convenience over `urllib.request`. Principle III's ladder puts stdlib above a new dependency, and
FR-029's real requirements — per-region requests, retry, timeout, rate-limit handling, caching, no
secret logged — are all policy this feature implements itself regardless of which client carries the
bytes.

**Alternatives considered**: `requests` (rejected — a new declared dependency for one call, and its
presence in this environment is incidental); `httpx` (rejected — same, plus its async surface is
unused since the pipeline is synchronous and per-region); the provider's own SDK (rejected — a
vendor SDK would couple the adapter to one provider, which is precisely what FR-029's pluggability
forbids, and it would be a heavier dependency than the raw call).

---

## R5 — Region extraction: connected components for the regions, contours for the polygons

**Decision**: `regions.py` binarises via Spec 1's `normalize_mask`, then uses
`cv2.connectedComponentsWithStats` for the component set, area and bounding boxes; merges components
whose gap is within the configured merge distance by **union-find over a distance test on expanded
boxes**, then recomputes each merged group's bbox; and derives each region's polygon with
`cv2.findContours` restricted to that group's mask, simplified with `cv2.approxPolyDP` at a
configured epsilon. Both the bbox and the polygon are stored (FR-018).

**Rationale**: FR-014 permits either connected components or contours, and the two are not
interchangeable here: components give the area and the bounding box cheaply and unambiguously, while
contours give the polygon FR-018 requires. Using each for what it is good at is less code than
deriving both from one of them, and it keeps the merge step — which FR-015 makes configurable and
which must be deterministic (FR-019) — operating on simple geometry rather than on contour trees.
`cv2` is already a declared dependency, so this adds nothing. The merge distance of zero (or merging
disabled) falls out of the same code path as the strict-inequality boundary, which is what US2
scenario 2 requires.

**Alternatives considered**: Contours only, with `boundingRect` for the box (rejected — contour
hierarchy handling for nested furigana is more code than the component path, and area comes free
from components); components only, with the bbox serving as the "polygon" (rejected — FR-018 says
*both*, and a rectangle is not a polygon of the text area); a watershed or morphological-closing
merge (rejected — closing changes the mask's area and therefore the region's geometry, making the
merge distance expressible only as a kernel size rather than as a distance).

---

## R6 — The bundled default font is `DejaVuSans.ttf`, committed under `assets/fonts/`

**Decision**: Commit `DejaVuSans.ttf` at `assets/fonts/DejaVuSans.ttf` with its licence beside it as
`assets/fonts/DejaVuSans.LICENSE.txt`, and make it the configured default `font_family`. The
rendering path resolves the font **by file path from configuration**, never by family name through a
system font lookup, so FR-039's "the default rendering path MUST NOT depend on system-installed
fonts" holds on a machine with no fonts installed.

**Rationale**: Measured, not assumed. DejaVuSans.ttf is 756,072 bytes, is already on every machine
that has matplotlib (so it can be copied into the repository without a download step), is under a
permissive licence that permits redistribution with the licence text, and its glyph coverage was
checked with `fontTools`' `getBestCmap()`: **Vietnamese 9/9** of the probe set (`ếệăđơưạảấ`), Latin
6/6, and **CJK 0/8** (`日本語のテキスト`). That coverage profile is exactly right for this feature —
the render target is Vietnamese or English, and the source Japanese never reaches the renderer as
text to draw (it is the OCR input, and it is only carried forward on a translation failure, where the
default policy still renders it). The one case the CJK gap touches is a translation-failed region
rendered with its Japanese OCR text; the renderer records that as a rendering result and the
documented fallback chain applies, rather than pretending the glyphs exist. Committing the font is
also what makes SC-013's no-vendoring rule consistent: SC-013 forbids model source and weights, and a
font is neither — it is a build input under an open licence, and it is the only binary this feature
adds to the tree.

**Alternatives considered**: Depending on a system font such as `arial.ttf` (rejected — FR-039
forbids it for the default path, and it is absent on the CI and POSIX machines this project also
runs on); downloading a font at first use (rejected — the default path would then need the network,
and FR-060 requires the fixture suite to run without it); a CJK-capable font such as Noto Sans JP
(rejected — tens of megabytes for glyphs the render target never uses, and it would not help the
failure case, since a missing translation is a policy outcome, not a font problem); loading the font
from matplotlib's package directory at runtime (rejected — a runtime dependency on another package's
internal layout, and it would silently change the rendered output if matplotlib ever changed its
bundled fonts).

---

## R7 — FR-044's optional external rendering module is declined

**Decision**: No external rendering module is adopted. Rendering is implemented in this feature with
Pillow (`ImageDraw.text` with a TrueType font, `font.getbbox`/`getlength` for measurement,
`textbbox` for line layout, and `stroke_width`/`stroke_fill` for the outline).

**Rationale**: FR-044 says an external module *MAY* be used and *MUST* be wrapped behind an adapter
with its version recorded. It is a permission, not a requirement. Pillow covers every rendering
capability FR-039 names — font family, size, colour, outline/stroke, a background box, word wrap,
line spacing, horizontal and vertical alignment, and text direction — plus the measurement
primitives FR-040's fitting ladder needs. Adding a second rendering dependency on top of Pillow would
buy nothing and would add a version to record and a licence to check. The adapter seam still exists
in the design (the renderer is reached through a protocol), so adopting one later is a new
implementation of an existing protocol, not a restructuring — which is the only condition under
which declining is safe.

**Alternatives considered**: A dedicated text-layout library (rejected — FR-040's ladder is
measurement plus a loop, which Pillow provides); `cv2.putText` (rejected — no TrueType, no wrapping,
no measured layout, and no outline; it cannot satisfy FR-039 at all); a headless browser or a
document renderer (rejected — enormously heavier than the requirement, and it would need a browser
runtime on the CPU-only target).

---

## R8 — Determinism discipline for the local stages

**Decision**: Extraction, OCR and rendering are deterministic by construction, enforced by four
rules: (a) **no wall-clock timestamp** is written into any per-page metadata field — the run ID is
the only time-identifying value (FR-058 with SC-012, mirroring Spec 3's FR-023 decision); (b) every
iteration over a set, dict or filesystem listing is **sorted by an explicit key** before it can
affect output order; (c) all thresholds and counts come from configuration, never from a computed
default that could vary with input; (d) floats that reach metadata are rounded to a configured number
of decimal places. Elapsed-time fields are the sole exception and are excluded from byte-comparison,
exactly as Spec 3's determinism scenario specifies.

**Rationale**: FR-058 requires determinism for the local stages and US3 scenario 4 and US2 scenario 4
require it specifically for OCR text and for region IDs. The two realistic ways determinism breaks
here are a timestamp that differs between runs and an unordered container that happens to iterate
differently — neither is visible in a single run and both are trivially avoided if the rule is
stated. Stating it once, here, is cheaper than discovering it in a byte-comparison test later.
Translation is deliberately outside this: FR-058 records provider nondeterminism as a *provider
property*, so the metadata records the provider's determinism as a declared property and the
determinism check never spans the translation stage.

**Alternatives considered**: Relying on the tests to catch nondeterminism (rejected — a
byte-comparison test tells you *that* two runs differ, not which of the four rules was broken, and
the fix is the same in all four cases); seeding a global RNG (rejected — nothing here is random; the
sources of variation are clocks and container order, which seeding does not address).

---

## R9 — Reading order and stable region ordering

**Decision**: Reading order is computed from configuration, defaulting to manga order —
**right-to-left, then top-to-bottom** — implemented as a total order on regions: sort by the
region's horizontal centre **descending**, then by vertical centre **ascending**, then by bounding-box
area descending, then by the component's top-left coordinate ascending as a final tie-break. The
`region_id` is derived from the region's **1-based index in that total order** under the active
configuration (FR-019), so it is stable and deterministic for the same mask and configuration and
changes only when the configuration or the region set changes.

**Rationale**: FR-019 requires `region_id` to be stable and derived from reading-order position, and
US2 scenario 4 requires identical IDs across repeated runs. A total order is what makes the
derivation well-defined: a partial order (right-to-left by column, say) leaves ties, and a tie broken
by container iteration order is the nondeterminism R8 forbids. The two extra tie-break keys are not
decoration — furigana stacked over a kanji, or two bubbles at the same vertical centre, produce
genuine ties in real pages, and area-then-position resolves them the same way every run. The default
direction is configurable because FR-016 names reading order as a configuration point, and because a
non-manga layout would need left-to-right.

**Alternatives considered**: Sorting by `cv2.connectedComponentsWithStats` label order (rejected —
label order is scan order, which is left-to-right and therefore wrong for manga, and it is an
implementation detail rather than a documented rule); a full geometric line-grouping algorithm
(rejected — more code than the requirement, and the merge step of FR-015 already handles the case
where components belong together); a content hash as the `region_id` (rejected — stable across runs
but not derived from reading-order position, so it fails FR-019 and would not survive a
configuration change in the documented way).

---

## R10 — Translation cache: content-addressed, outside every run tree, bypassable

**Decision**: The cache lives at `cache/translation/` — a single shared directory **outside** every
run directory (FR-029) and already gitignored by `.gitignore`. The key is a SHA-256 over the
canonical JSON of `(text, target_language, provider, model)`; the stored value is a JSON record
carrying the translated text, the provider/model/version, the cache key and the timestamp of the
original fetch. `configs/translation.json` exposes a `bypass` switch that forces fresh provider calls.
Each page's metadata records, per region, whether the translation was **served from cache** or
**fetched live** (FR-029).

**Rationale**: Content addressing is what makes the cache correct rather than merely fast: the key
contains every input that can change the answer, so a hit is always the right answer for that
request, and no invalidation logic is needed. The key deliberately excludes the page, the region and
the run, because the same Japanese string in two pages should be fetched once — that is the entire
point of a shared cache, and it is why the cache cannot live inside a run directory. Storing the
fetch timestamp in the record is what makes the cache auditable after the fact without a wall clock
in any page metadata, which keeps R8's determinism rule intact. `cache/` is already in `.gitignore`,
so no cache file can reach the repository.

**Alternatives considered**: A per-run cache (rejected — FR-029 resolved to shared-by-default, and a
per-run cache re-fetches every repeated string on every run); a cache keyed by page and region
(rejected — same string in two pages would be fetched twice, and the key would be sensitive to
region renumbering); a TTL or an expiry policy (rejected — nothing in the spec asks for
invalidation, and a TTL would make a run's output depend on when it ran, which is precisely the
nondeterminism R8 exists to prevent); a database or a library such as `diskcache` (rejected — a
directory of JSON files keyed by hash is the whole requirement, and the ladder puts stdlib above a
new dependency).
