---
name: Precision Research Workbench
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#45464d'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#76777d'
  outline-variant: '#c6c6cd'
  surface-tint: '#565e74'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#131b2e'
  on-primary-container: '#7c839b'
  inverse-primary: '#bec6e0'
  secondary: '#0051d5'
  on-secondary: '#ffffff'
  secondary-container: '#316bf3'
  on-secondary-container: '#fefcff'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#271901'
  on-tertiary-container: '#98805d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dae2fd'
  primary-fixed-dim: '#bec6e0'
  on-primary-fixed: '#131b2e'
  on-primary-fixed-variant: '#3f465c'
  secondary-fixed: '#dbe1ff'
  secondary-fixed-dim: '#b4c5ff'
  on-secondary-fixed: '#00174b'
  on-secondary-fixed-variant: '#003ea8'
  tertiary-fixed: '#fcdeb5'
  tertiary-fixed-dim: '#dec29a'
  on-tertiary-fixed: '#271901'
  on-tertiary-fixed-variant: '#574425'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-lg:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 1.5rem
    fontWeight: '600'
    lineHeight: 2rem
    letterSpacing: -0.02em
  headline-md:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 1.125rem
    fontWeight: '600'
    lineHeight: 1.5rem
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 0.9375rem
    fontWeight: '600'
    lineHeight: 1.375rem
    letterSpacing: -0.01em
  body-default:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 0.875rem
    fontWeight: '400'
    lineHeight: 1.375rem
  body-compact:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 0.8125rem
    fontWeight: '400'
    lineHeight: 1.25rem
  label-default:
    fontFamily: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      sans-serif
    fontSize: 0.75rem
    fontWeight: '500'
    lineHeight: 1rem
    letterSpacing: 0.02em
  code-default:
    fontFamily: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation
      Mono', monospace
    fontSize: 0.8125rem
    fontWeight: '400'
    lineHeight: 1.25rem
  code-compact:
    fontFamily: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation
      Mono', monospace
    fontSize: 0.6875rem
    fontWeight: '500'
    lineHeight: 0.875rem
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  gutter: 0.75rem
  margin: 1rem
  space-2xs: 0.125rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1rem
  space-xl: 1.5rem
  space-2xl: 2rem
---

## Brand & Style

This design system establishes a high-density, analytical desktop environment tailored for computer vision researchers, digital archivists, and computational linguists evaluating manga text extraction pipelines.

The core aesthetic combines utilitarian minimalism with technical instrumentation:
- **Tone:** Objective, measured, quiet, and reliable. The tool recedes completely so high-contrast manga line art and overlaid segmentation masks command primary visual attention.
- **Visual Tenets:** Strict 1px division lines, absence of decorative gradients or glassmorphism, neutral slate/zinc surfaces, calibrated categorical data palettes, and monospace typography reserved for coordinates, hashes, and evaluation metrics.
- **Interaction Feel:** Immediate, low-latency, deterministic. Affordances rely on border color transitions, background tint shifts, and discrete state tokens rather than motion-heavy flourishes.

## Colors

The palette establishes a neutral backdrop that maintains perceptual accuracy for scanned manga artwork across both greyscale tones and colored overlays.

### Categorical Method Identifiers
The 4 segmentation models must strictly use designated categorical tokens across bounding boxes, polygon masks, legend tags, confusion matrices, and benchmark tables:
- **Classical Baseline:** `#3b82f6` (Blue)
- **Manga-Text-Segmentation:** `#10b981` (Emerald)
- **comic-text-detector:** `#f59e0b` (Amber)
- **UNet++ EfficientNetV2:** `#8b5cf6` (Violet)

### Status Signaling
Status states (Inference Running, Converged, Out of Memory, IoU Degraded) must pair their status color with a shape-distinctive icon (e.g., checkmark, triangle warning, octagonal error) to preserve WCAG accessibility without relying solely on color variance.

## Typography

This system operates entirely on native system font stacks to ensure instantaneous font rendering, true offline reliability in isolated compute environments, and zero runtime overhead.

- **Prose & Controls:** Use `system-ui` for layout controls, modal dialogues, parameter titles, and workflow documentation.
- **Data & Metadata:** Use `ui-monospace` for run IDs, bounding box coordinate arrays `[ymin, xmin, ymax, xmax]`, IoU scores, OCR character confidence, model latency counters, and dataset paths.
- **Vertical Rhythm:** Line heights match a 4px/8px baseline grid to keep tabular parameter lists tight and prevent vertical drift during multi-column data reviews.

## Layout & Spacing

Designed for dense, information-rich multi-pane desktop layouts (split-pane inspection, comparison matrix, segmentation canvas, OCR/translation terminal).

### Grid & Panes
- **Layout Architecture:** Fixed/collapsible multi-column workbench structure. A primary sidebar (280px fixed width), a center interactive canvas (fluid stretch), and a secondary telemetry/translation panel (360px fixed width).
- **Rhythm:** Spacing follows a 4px baseline rhythm. Padding within interactive controls, toolbar icon arrays, and tabular rows is kept compact (`space-xs` to `space-md`) to maximize vertical manga viewport space on standard 1080p and 1440p displays.

## Elevation & Depth

Visual hierarchy is constructed via crisp surface stacking and border delineations, deliberately excluding drop shadows or diffuse blur filters:

- **Surface Tiers:** Three discrete planes establish hierarchy: Canvas/Window backdrop (`bg-canvas`), Panel surface (`bg-surface`), and Sub-panels/Toolbars (`bg-subtle`).
- **Low-Contrast Outlines:** Every container, viewport pane, inspector card, and contextual menu relies exclusively on a clean `1px solid` border (`border-subtle` or `border-default`).
- **Overlays & Modals:** Floating inspection overlays, tooltips, and flyout menus employ a sharp `border-strong` stroke with a micro-offset `0 2px 4px rgba(0, 0, 0, 0.06)` shadow for edge separation without breaking the technical workbench aesthetic.

## Shapes

The interface balances sharp technical precision with usability via consistent 8px corners:
- Standard containers, viewports, panel frames, input controls, and modal shells use an 8px radius (`0.5rem`).
- Internal compact tags, pill indicators, and mini-buttons use a 4px radius (`0.25rem`) to prevent roundness collision inside tight data grids.
- Pure circular bounds (`rounded-full`) are strictly reserved for toggle switches and status dot nodes.

## Components

### Buttons & Toolbars
- **Standard Button:** Height 32px, radius 8px, padding 0 12px. Border `1px solid border-default`. Background `bg-surface`, text `text-primary`. Hover state switches background to `bg-subtle`.
- **Method Toggle Button:** Displays model label alongside a 8px colored swatch matching its categorical token. Selected state adds an active `2px` left border tint in that categorical color.
- **Toolbar Icon Buttons:** 28x28px square, 8px radius, borderless until hover. Active tool states maintain a flat `bg-subtle` and `1px solid border-default`.

### Chips & Method Badges
- **Method Badge:** Monospace font (`code-compact`), height 20px, 4px radius. Uses a 10% opacity tint of the method's categorical color for background, a 40% border of the same hue, and 100% saturation for the text label.
- **Metric Pill:** Monospace font displaying performance telemetry (e.g., `IoU: 0.884`, `lat: 42ms`), neutral slate styling, 4px radius.

### Input Fields & Parameter Steppers
- **Technical Inputs:** Height 28px, 8px radius, `1px solid border-default`, monospace text for threshold values, coordinates, and learning rates. Focus state creates a sharp `1px solid text-primary` ring without diffuse outline glows.
- **Sliders:** 4px flat track in `border-default`, square-rounded 12px thumb with neutral styling.

### Cards & Inspector Panels
- **Inspection Card:** Bound by `1px solid border-subtle` with zero elevation. Contains a 28px header section separated by a horizontal divider line.
- **Translation Block Container:** Split-cell layout. Left cell holds original vertical Japanese OCR transcript in `code-default`; right cell displays model translation with alignment score indicator.

### Canvas Overlays & Bounding Boxes
- **Segmentation Polygons:** Stroke width 1.5px using the active method's categorical color. Fill uses 12% alpha opacity of the same color. Hover/selection increases stroke to 2px and fill to 25% opacity.