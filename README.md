# Materia 1.2

*Analyse · Visualise · Understand* — formerly LabStudio

A desktop toolkit for shape memory alloy (NiTi) characterisation data: import → plot → analyse → export,
plus an ImageJ-style micrograph workspace for SEM and light optical microscopy.

## Install & run

1. Install **Python 3.10 or newer** from python.org (on Windows tick *"Add Python to PATH"*).
2. **Extract the zip** (Windows: right-click ▸ Extract All), then double-click
   **Start Materia (Windows).bat** or **Start Materia (Mac).command**.
   The Materia loading screen appears; on the first start it sets up a private environment (`.venv`) with a
   progress bar (a few minutes, needs internet). On Windows it then creates a **Materia** shortcut with the
   Materia icon in the folder and on the Desktop — use that from then on (no console window).

Manual alternative:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python run_labstudio.py   # or: python launcher.pyw [file.csv | session.lstudio | image.tif]
```

Try **File ▸ Load sample NiTi data** to see every feature on the synthetic NiTi files in `sample_data/`.

## Updates

Materia updates itself. Each start, the loading screen checks GitHub (Eyto88/materia) for a newer version and
offers **Update now / Later / Skip this version**, showing what's new. Only files that changed are replaced;
your settings, presets, styles, sessions and any files of your own are kept, and packages are reinstalled only if
needed. Inside the app: **Help ▸ Check for updates…** and **Help ▸ Restore previous version…**.

## New in 1.2.1

- Automatic updates (see above).

## New in 1.2

- Materia branding: loading screen with progress instead of a console, Materia icon on the taskbar,
  window and Desktop shortcut.

## New in 1.1

- **No silent wrong answers**: amber warnings for guessed units, default/mismatched DSC heating rates,
  truncated peaks, heating/cooling ΔH mismatch, incomplete cycles, impossible values, print-dpi calibration,
  low grain statistics. Warnings travel into every CSV.
- **Sessions** (`.lstudio`): datasets, figures, analyses and micrographs in one file; recent sessions.
- **Provenance** on every export: version, parameters, source file + SHA-256.
- **Preferences**: significant figures, CSV precision, default figure template. Help ▸ Show log / Report a problem.
- **Import**: Excel (.xlsx/.xls/.ods) with sheet picker, spelled-out units, case-safe SI prefixes (MJ ≠ mJ),
  all common DSC heating-rate keys, analysis defaults stored in presets.
- **Plotting**: segment-aware integration, style templates (Thesis / Journal / Slides / your own),
  multi-panel figures with (a)(b)(c) labels, palettes & colormaps with colorbars, waterfall offsets,
  mean ± SD of repeats, non-overlapping labels, preview at export size, dataset rename/reorder/remove.
- **Analysis**: robust noisy-DSC peak finding, overlapping R/M peaks (split or combined), editable peak limits,
  DSC segment selection/splitting, compression-signed cyclic data, compliance correction, fatigue trend fits,
  **Batch** tab (many samples → one table, metric vs parameter, mean ± SD), **eCaloric** tab
  (Clausius–Clapeyron dσ/dT → ΔS, L with DSC cross-check; measured ΔT, COP and specific cooling power per cycle).
- **Micrographs**: trusted calibration sources only (shown on screen), ImageJ-identical perimeter/circularity,
  scale bars and labels excluded from particle analysis, µ renders in burned-in scale bars, scale-bar presets,
  background processing (8k × 8k grain size ≈ 15 s).
- Fits a 1280 × 720 laptop screen; long jobs run in the background.

## What it does

### Data import (CSV / TXT from DSC, tensile, compression, cyclic machines)
- Auto-detects encoding, delimiter, decimal comma, instrument preamble, header row and units row
  (Netzsch, TA, Mettler, Instron Bluehill, Zwick, Shimadzu and generic exports).
- Import dialog: for each column you confirm the **quantity** (force, displacement, temperature, heat flow…)
  and the **unit it was recorded in**; values are converted to consistent units on import:
  N, mm, mm², MPa, %, °C, s, mW, W/g, J/g, MJ/m³.
- Enter specimen geometry (area, round Ø or w × t, gauge length) or DSC mass / heating rate.
- **Save the mapping as a preset** per machine so the next file imports in one click.

### Plot & Analyse tab
- Overlay any X/Y columns from any datasets; secondary Y axis; per-trace colour, line, marker, label.
- Customise title, labels, limits, log scales, fonts, ticks, grid, legend; style presets
  (Default / Publication / Presentation); show axes in other units without changing the data.
- Interactive tools: data cursor (snaps to points), peak finder, span selection, integration
  (whole curve or a span, with optional linear baseline, physically meaningful units),
  linear fit / slope, span statistics. Everything goes to a results table → CSV / clipboard.
- Export PNG (150/300/600 dpi, size in cm, transparent), SVG or PDF.
- **Analysis dock**
  - *DSC*: splits heating/cooling, As/Af/Ms/Mf/Rs/Rf by tangent onset, peak temperatures,
    latent heat (J/g), thermal hysteresis, estimated adiabatic ΔT (L/cp).
  - *Tensile / compression*: stress–strain from force–displacement, E, 0.2 % offset yield, UTS,
    fracture strain, toughness, σ_SIM for superelastic NiTi.
  - *Cyclic*: cycle splitting, per-cycle dissipated energy (loop area), residual and recoverable strain,
    upper/lower plateau stresses, stress hysteresis, functional-fatigue summary vs cycle number.

### Micrographs tab (ImageJ-style)
- Opens TIFF (8/16-bit, multipage), PNG, JPG, BMP; reads pixel size from FEI/Thermo and Zeiss SEM TIFF
  metadata; detects and crops the SEM data bar.
- Set scale from a scale bar; measure lengths, angles, areas, perimeters, intensities; line profiles;
  point counter; ROI manager; results table.
- Brightness/contrast, gamma, CLAHE, filters, FFT bandpass, rolling-ball background subtraction,
  crop/rotate/flip/resize, undo/redo.
- Threshold (manual + Otsu, Li, Yen, Triangle, IsoData…), binary operations, watershed.
- Analyze particles (area, ECD, circularity, aspect ratio, solidity, Feret…), size histograms.
- Grain size: ASTM E112 linear intercept and Jeffries planimetric (G number), grain-area distribution.
- Phase fraction (threshold or multi-level Otsu), histograms, burned-in scale bars, annotated export.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q            # 739 tests; LABSTUDIO_SLOW=1 adds the 8k-image performance test
```

## Layout
```
labstudio/core   pure-Python units, data model, import, analysis, image processing (unit-tested)
labstudio/gui    PySide6 user interface
sample_data/     synthetic NiTi example files
tools/           make_sample_data.py
```
