# LabStudio — architecture & module contracts

Desktop app (Python 3.10+, **PySide6** + **matplotlib** (`backend_qtagg`) + numpy/scipy/pandas/scikit-image/tifffile)
for an MEng project on elastocaloric NiTi shape memory alloys. It loads CSV/TXT exports from DSC,
tensile (failure), compression and cyclic (superelastic) tests, plots them with full customisation,
lets the user interact (cursor read-out, peaks, integration), runs materials-specific analysis, exports PNG,
and has an ImageJ-style micrograph workspace for SEM / light optical microscopy (LOM).

No other GUI toolkits. No network access at runtime. Everything must work offline on Windows/macOS.

## Golden rules
1. **Units**: all values stored in a `Dataset` are already in canonical units (`labstudio/core/units.py`
   `CANONICAL`): force N, displacement/length mm, area mm², stress MPa, strain %, temperature °C, time s,
   heat flow mW, specific heat flow W/g, energy J, specific energy J/g, energy density MJ/m³, heating rate K/min, voltage µV.
   Every number shown to the user carries its unit.
2. `labstudio/core/units.py` and `labstudio/core/dataset.py` are owned by the lead. **Do not edit them.**
   If you need a helper, put it in your own module; mention any wished-for change in your final report.
3. `core/` modules are pure Python (no Qt imports) and unit-tested with pytest. `gui/` modules hold Qt code.
4. GUI must run headless for tests: `QT_QPA_PLATFORM=offscreen`. Never block on a modal dialog in tests.
5. Only edit files you own (listed below). Write tests in `tests/test_<yourmodule>.py`.
6. Type hints + docstrings on public functions. Keep the UI clean: QSplitter layouts, collapsible
   QGroupBoxes / QToolBox for option panels, tooltips explaining each parameter.

## Shared model (already written — read it)
- `core/units.py`: `convert(values, from_unit, to_unit, dimension=None)`, `to_canonical`, `units_for(dim)`,
  `dimensions()`, `CANONICAL`, `DIMENSION_LABELS`, `parse_header(header) -> (name, unit, dimension)`, `axis_label`.
- `core/dataset.py`: `Channel(name, dimension, unit, original_unit)`, `Dataset(name, data: DataFrame, channels,
  kind, source_path, metadata, default_x, default_y, id)` with `.values(col, unit=None)`, `.label(col)`,
  `.columns_of(dim)`, `.first_of(*dims)`, `.add_column(name, values, dimension, unit=None)`, `.subset(mask)`,
  `Dataset.from_arrays(name, {col: (values, dim, unit)}, kind=...)`; `DatasetStore` with `add/remove/get/
  subscribe(fn)/notify_changed()/iter`.
- Dataset kinds: `generic | dsc | tensile | compression | cyclic`.
- Metadata keys (canonical units): mechanical → `area_mm2`, `gauge_length_mm`, optional `diameter_mm`,
  `width_mm`, `thickness_mm`; DSC → `mass_mg`, `heating_rate_K_per_min`.

## Modules and owners

### A — Data import (`core/data_io.py`, `gui/import_dialog.py`, `tests/test_data_io.py`)
- `read_table(path) -> RawTable` dataclass: `df` (raw strings→numeric DataFrame), `headers: list[str]`,
  `meta_lines: list[str]` (instrument preamble), `delimiter`, `decimal`, `encoding`, `header_row`.
  Robust sniffing: encodings utf-8/utf-8-sig/utf-16/cp1252/latin-1; delimiters `, ; \t |` and whitespace;
  decimal comma; preamble/metadata lines before the header (e.g. Netzsch `#` lines, Instron header block,
  TA Instruments); a units row directly under the header (e.g. `Time,Extension,Load` / `(s),(mm),(kN)`);
  trailing junk/non-numeric rows; thousands separators. Also parse `key: value` / `#KEY:value` metadata
  pairs into `RawTable.meta: dict` (mass, heating rate, sample name if present).
- `ColumnSpec(source: str, name: str, dimension: str, unit: str, include: bool=True)`.
- `guess_mapping(raw) -> list[ColumnSpec]` using `units.parse_header` (+ units row).
- `guess_kind(raw) -> str`.
- `build_dataset(raw, specs, name, kind, metadata) -> Dataset` (converts to canonical units; sets default_x/y
  sensibly per kind: DSC → temperature vs heat flow; mechanical → strain vs stress if present else
  displacement vs force).
- Presets: `save_preset(path_or_name, specs, kind)`, `load_preset(...)`, `list_presets()` stored as JSON in
  `~/.labstudio/presets/` so a mapping for a given machine can be reused in one click.
- `load_file(path, preset=None) -> Dataset` convenience (auto-guess, no GUI).
- `export_dataset_csv(ds, path)` with unit-bearing headers.
- `gui/import_dialog.py`: `ImportDialog(path, parent=None)` (QDialog). Shows preamble, preview table (first
  200 rows), per-column row: include checkbox, name edit, dimension combo, unit combo (units of that
  dimension, pre-filled from guess). Kind combo. Metadata form that changes with kind: mechanical —
  cross-section entry mode (area directly / round Ø / rectangular w×t) with unit combos, gauge length;
  DSC — mass, heating rate (pre-filled from file meta if found). Preset combo + save preset button.
  `.dataset() -> Dataset | None` after `exec()`. Also `ImportDialog.import_files(paths, parent) -> list[Dataset]`
  helper that opens one dialog per file (with "apply to all remaining" checkbox).

### B — Plotting (`gui/plot_widget.py`, `gui/plot_style.py`, `tests/test_plot_widget.py`)
`PlotPanel(QWidget)` = the whole "Plot" tab: left = dataset/trace manager, centre = canvas + matplotlib
NavigationToolbar + tool toolbar, right = customisation panel, bottom = results table.
Public API the analysis module relies on (implement exactly):
```python
class PlotPanel(QWidget):
    resultsAdded = Signal(list)                      # list[dict] rows
    def __init__(self, store: DatasetStore, parent=None)
    figure: matplotlib.figure.Figure; canvas; ax  (primary Axes); ax2 (twin y, created lazily)
    def add_trace(self, ds: Dataset, x: str, y: str, label: str | None = None, *, secondary_y=False,
                  **style) -> "Trace"            # style: color, linestyle, linewidth, marker, markersize, alpha
    def remove_trace(self, trace) ; def clear(self); traces: list[Trace]
    def selected_trace(self) -> Trace | None
    def add_marker(self, x, y, text: str = "", *, color=None, marker="o") -> artist
    def add_vline(self, x, text="", **kw) ; def add_hline(self, y, text="", **kw)
    def add_line(self, xs, ys, label=None, **kw)    # overlay (e.g. tangent, modulus fit)
    def fill_between(self, x, y1, y2=0, **kw)        # shade integration areas
    def clear_overlays(self)                         # remove markers/lines/fills added by the API
    def add_results(self, rows: list[dict])          # append rows to results table; emits resultsAdded
    def redraw(self)
    def export_png(self, path: str, dpi: int = 300, transparent=False, size_in=None)
    def current_span(self) -> tuple[float, float] | None   # x-range selected with span tool
```
`Trace`: dataclass with `ds`, `x`, `y`, `label`, `line` (Line2D), `secondary_y`, `.xy() -> (x, y arrays)`.
Features:
- Trace manager: pick dataset, X column, Y column (combos show `name (unit)`), Add; list of traces with
  visibility checkbox, delete, style editing (colour picker, line style, width, marker, marker size, label),
  move to secondary Y axis. Auto axis labels from channel labels. Display-unit override per axis
  (e.g. show stress in GPa, temp in K) via `units.convert` — values in the store stay canonical.
- Customisation panel: title; x/y(/y2) labels; limits (auto or manual); linear/log; invert axis; tick
  label size; label size; font family; grid major/minor; legend on/off, location, columns, frame, font size;
  figure background; style presets ("Default", "Publication (serif, 300 dpi, inward ticks)", "Presentation
  (large fonts)"); text annotation tool. Changes apply live.
- Interactive tools (toolbar, mutually exclusive): **Data cursor** (snaps to nearest point on nearest trace,
  shows x,y with units in a tooltip-style annotation + status bar; click to pin a point and add it to results);
  **Peak finder** (scipy.signal.find_peaks on selected trace, with prominence/min distance/sign
  (peaks/valleys) controls; marks & lists x, y, prominence, FWHM); **Span select** (matplotlib SpanSelector);
  **Integrate** — whole trace or selected span, baseline none / linear between span ends; shades area;
  reports area with unit = y_unit·x_unit (e.g. MPa·% → also give MJ/m³ when x=strain %, y=stress MPa:
  1 MPa·% = 0.01 MJ/m³; for DSC heat flow vs time give J; W/g vs time give J/g); **Slope/linear fit** on span
  (slope with units, intercept, R²); **Min/Max/mean** stats on span.
- Results table: columns Tool | Trace | Quantity | Value | Unit ; copy to clipboard, export CSV, clear.
- Export: PNG (dpi choice 150/300/600, transparent, figure size in cm), also SVG/PDF via same dialog.
  "Copy figure to clipboard".
- Put matplotlib rcParam presets in `gui/plot_style.py`.

### C — Analysis (`core/analysis.py`, `gui/analysis_panel.py`, `tools/make_sample_data.py`, `tests/test_analysis.py`)
Pure functions (return dataclasses with values in canonical units, plus fields for plotting overlays):
- General: `find_peaks_xy`, `integrate_xy(x, y, xmin=None, xmax=None, baseline="none"|"linear")`,
  `linear_fit(x, y, xmin, xmax)` → slope, intercept, r2.
- DSC: `split_dsc_segments(T, t=None)` → heating/cooling masks (from sign of dT/dt with smoothing, ignores
  isotherms); `analyse_dsc_segment(T, hf, heating_rate_K_per_min, mass_mg=None, direction, baseline="linear",
  prominence=...)` → list of `DSCPeak(start_T, onset_T, peak_T, endset_T, end_T, enthalpy_J_per_g, ...)` with
  onset/endset by tangent construction at inflection points intersecting the baseline; enthalpy =
  ∫(hf−baseline)dt (convert T-axis to time via heating rate) / mass. Map to SMA nomenclature: heating →
  As, Af (and R-phase if two peaks); cooling → Rs/Rf, Ms, Mf. Handle heat flow in mW (needs mass) or W/g.
  Handle endo-up/endo-down sign convention (parameter + auto-detect). Also `delta_T_ad_estimate(latent J/g,
  cp=0.45 J/gK, T)` ≈ L/cp for elastocaloric relevance.
- Mechanical: `stress_strain(force_N, disp_mm, area_mm2, gauge_mm, compression=False)`;
  `tensile_properties(strain_pct, stress_MPa, modulus_range=None)` → E (GPa) by linear fit over auto-detected
  linear region or user range, 0.2 % offset yield, UTS, strain at UTS, fracture strain (detect final drop),
  toughness (MJ/m³), plus overlay lines (modulus line, offset line). For superelastic NiTi also detect the
  forward transformation plateau onset (σ_SIM) on loading.
  `compression_properties` (same with compressive sign handling; yield, max stress, strain at max).
- Cyclic/superelastic: `split_cycles(strain, stress, cycle=None, time=None)` (use a cycle-number column if
  present, else detect load reversals by strain peaks with prominence); `cycle_metrics(...)` per cycle →
  max stress, max strain, residual (unrecovered) strain, recoverable strain, hysteresis loop area =
  dissipated energy (MJ/m³), loading work, upper plateau σ_ms (forward), lower plateau σ_as (reverse), stress
  hysteresis Δσ, plateau measured at mid-plateau strain; `fatigue_summary` table across cycles (functional
  fatigue: change in residual strain, Δσ, dissipated energy vs cycle number). Also export per-cycle as new
  datasets.
- `gui/analysis_panel.py`: `AnalysisPanel(plot_panel: PlotPanel, store: DatasetStore, parent=None)` QWidget
  (will be docked next to the plot). QTabWidget: DSC | Tensile / Compression | Cyclic. Each: dataset combo
  (filtered by kind, but allow any), column combos (pre-selected by dimension), parameters with tooltips,
  Run button → draws overlays on the plot through the PlotPanel API (markers at onsets/peaks labelled As,
  Af…, tangent lines, shaded enthalpy areas, modulus/offset lines, plateau lines), writes rows via
  `plot_panel.add_results`, and shows a summary table with "Export CSV". Mechanical tab: "Compute stress–
  strain" button adds `Stress`/`Strain` columns to the dataset from force/displacement + metadata (asks for
  area/gauge if missing) then plots stress vs strain. Cyclic tab: plot cycles coloured by cycle number
  (viridis) and a second button "Plot fatigue summary" which creates a derived Dataset (cycle vs metric) and
  adds it to the store.
- `tools/make_sample_data.py`: writes realistic synthetic NiTi files to `sample_data/`:
  `NiTi_DSC_Netzsch.txt` (Netzsch-style `#` preamble with mass, rate; `##Temp./°C;Time/min;DSC/(mW/mg)`
  semicolon, heating+cooling, −80→100 °C, two-step B2→R→B19' on cooling, single peak heating, ~20 J/g),
  `NiTi_tensile_failure_Instron.csv` (preamble, header + units row, Time/Extension (mm)/Load (kN), wire Ø
  1 mm gauge 50 mm, superelastic plateau ~450 MPa then hardening to UTS ~1300 MPa, fracture ~14 %),
  `NiTi_compression.csv`, `NiTi_superelastic_cyclic_50cycles.csv` (Cycle, Time (s), Strain (%), Stress (MPa),
  6 % strain, functional fatigue: residual strain accumulates, plateau stresses drop, hysteresis shrinks),
  `generic_xy.txt` (tab separated, decimal comma). Must be parseable by module A's auto-guess.

### D — Micrographs (`core/image_ops.py`, `core/micro_analysis.py`, `gui/image_workspace.py`,
`gui/image_viewer.py`, `tests/test_image.py`)
ImageJ-style workspace for SEM and LOM.
- Loading: tif/tiff (8/16-bit, multi-page → pick page), png, jpg, bmp. **Read pixel size from SEM TIFF metadata**
  when present: FEI/Thermo (`FEI_HELIOS`/`FEI_SFEG` tag 34682 ini text `PixelWidth=` in m), Zeiss (tag 34118
  `Image Pixel Size = 12.3 nm`), JEOL/Hitachi sidecar `.txt` (`$$SM_MICRON_BAR`, `PixelSize=`), standard
  resolution tags as fallback. Detect & offer to crop the SEM data bar (bottom info strip).
- Calibration: `Calibration(px_per_unit, unit)`; "Set scale" tool: draw line along scale bar, enter known
  length+unit (nm/µm/mm); global or per-image; show calibrated values everywhere.
- Measure: straight line (length, angle), multi-segment line, angle tool (3 points), rectangle, ellipse,
  polygon, freehand ROI → area, perimeter, mean/min/max/std intensity, centroid, Feret; line profile plot
  (intensity vs calibrated distance, with width); point counter (multi-point, counts). ROI manager list.
  Results table (ImageJ-like) with export CSV & clear.
- Adjust/Process: brightness/contrast/min-max with auto (saturated 0.35 %), gamma, invert, 8-bit conversion,
  histogram equalisation, CLAHE, Gaussian blur, median, mean, unsharp mask, sharpen, find edges (Sobel),
  FFT bandpass, background subtraction (rolling ball radius, light/dark background — skimage
  `restoration.rolling_ball`), crop to ROI, rotate (90/arbitrary), flip, resize. Undo/redo stack (≥20) and
  "Revert to original".
- Threshold: manual (two sliders with red overlay preview) + auto methods (Otsu, Li, Yen, Triangle, IsoData,
  Mean, Minimum, MaxEntropy-equivalent via skimage), dark/light objects. Binary: erode, dilate, open, close,
  fill holes, remove small objects, skeletonize, watershed separation (distance transform), outline.
- Analyze particles: size range (calibrated units²), circularity range, exclude edges, include holes →
  per-particle table (area, perimeter, equivalent circle diameter, major/minor axis, aspect ratio,
  circularity 4πA/P², solidity, Feret max/min, orientation, centroid) + summary (count, total area,
  area fraction, mean size), outline/label overlay, size-distribution histogram.
- Grain size: (1) **ASTM E112 / ISO 643 linear intercept** — overlay N horizontal/vertical/circular test lines
  on a grain-boundary binary (or from user threshold/edge detection), count intersections, mean intercept
  length l̄ (µm), ASTM G = −3.2877 − 6.6439·log10(l̄ in mm); (2) **planimetric (Jeffries)** grains per mm² →
  G = 3.321928·log10(N_A) − 2.954; (3) grain-area distribution from segmented grains (watershed / boundary
  inversion). Show test lines and intercept marks on the image.
- Phase fraction: by threshold(s) or multi-level Otsu (2–4 phases), area % of each phase with colour overlay;
  intensity histogram (with stats, log option, export).
- Overlays/annotation: burn-in scale bar (auto nice length, position, colour, font size, background box),
  text, arrows; export image as PNG/TIFF with or without overlays; export results CSV.
- `gui/image_workspace.py`: `ImageWorkspace(QWidget)` = the whole "Micrographs" tab; supports several open
  images (list or tabs), toolbar of tools, right-hand QToolBox panels (Adjust, Process, Threshold & Binary,
  Analyze particles, Grain size, Phase fraction, Histogram, Annotate), bottom results table. Viewer
  (`gui/image_viewer.py`) = QGraphicsView with zoom (wheel, fit, 100 %), pan, pixel read-out (x, y calibrated,
  value) in status line, ROI drawing items, overlays.
- `core/` functions must be pure (numpy in → numpy/dataclasses out) and tested on synthetic images
  (e.g. a Voronoi grain image with known grain count, discs of known area).

### Lead — integration (`labstudio/app.py`, `gui/main_window.py`, `run_labstudio.py`, README, packaging)
`MainWindow`: tabs "Plot & Analyse" (PlotPanel + AnalysisPanel in a dock/splitter) and "Micrographs"
(ImageWorkspace); menu File (Import data… Ctrl+O, Open image…, Export plot PNG, Export dataset CSV, Quit),
drag-and-drop of files (data → ImportDialog, images → Micrographs), dataset list context menu, About.

---------------------------------------------------------------------------------------------------
# v1.1 — fixes & improvements (contract for all v1.1 work)

Source of requirements: /home/claude/review/customer/customer_report.md and /home/claude/review/qa/rating_report.md
(+ repro scripts in /home/claude/review/qa/repro and customer test files in /home/claude/review/customer/messy).
Version is 1.1.0. Baseline is committed in git (`git diff` to review your changes; do NOT commit).

## Shared infrastructure (lead-owned; use, don't edit)
- `core/provenance.py`: `provenance(analysis, params, ds=None, source_path=None, extra=None) -> dict` and
  `provenance_lines(rec) -> list['# ...']`. EVERY analysis result and every results/CSV export must carry one:
  analysis name, all parameters (incl. auto-chosen ones), app version, timestamp, source file + SHA-256,
  plus a `warnings` list. CSV exports start with `provenance_lines(...)` comment lines.
- `core/session.py`: `.lstudio` session = datasets + panel states. Panels expose
  `get_state() -> dict` (JSON-serialisable; refer to datasets by `ds.id`) and
  `set_state(state, id_map: dict[int, Dataset])`. Optional `get_state_files() -> dict[str, bytes]` /
  `set_state_files(files)` for binary payloads (images). Panel keys: "plot" (PlotPanel), "analysis"
  (AnalysisPanel), "images" (ImageWorkspace).
- `gui/worker.py`: `run_task(parent, message, fn, on_done, on_error)` — run heavy pure computations off the GUI
  thread with a busy dialog; synchronous in tests (offscreen) so tests stay deterministic.
- `gui/app_settings.py`: `settings()` (QSettings-backed; isolated via env LABSTUDIO_HOME), `format_value(v)` =
  significant-figure formatting (Preferences "sig_figs", default 5). ALL copy-to-clipboard and table display
  of numbers uses format_value; CSV uses it unless settings().get("csv_full_precision").
- `core/dataset.py`: `DatasetStore.rename(ds, name)`, `.move(ds, index)`, `.index_of(ds)` added.
- `core/units.py`: `si_factor(dimension, unit)` added. (Workstream A now owns units.py for the prefix fix.)

## Warnings policy (the #1 theme of both reviews: no silent wrong answers)
Any result computed from a guessed/defaulted/suspicious input must say so: put human-readable strings in the
result's `warnings` list, show them in the GUI (amber status text / warning rows in results tables, never a
modal in tests), and include them in provenance. Examples: heating rate defaulted; header rate ≠ measured
rate; unit guessed from a spelled-out word; peak truncated at data edge; heating/cooling ΔH differ > 20 %;
cycle incomplete (no unloading); E outside 1–300 GPa; calibration from generic TIFF dpi; integrating a curve
that doubles back in x.

## Responsive layout rule
Every top-level panel must work in a 1280×720 window (and 1366×768 at 100 %, i.e. MainWindow must be able to
shrink to ≤ 1200×680): side panels in QScrollArea, minimum widths ≤ 260 px, collapsible sections, no fixed
large minimumSize. Test by constructing the widget, resize(1200, 680), and asserting
`widget.minimumSizeHint()` width ≤ 1200 / height ≤ 680 (for the docks/panels you own: ≤ 420 px wide for side
docks).
