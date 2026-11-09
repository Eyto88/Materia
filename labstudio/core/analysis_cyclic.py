"""Cyclic (superelastic) loading analysis and functional-fatigue summary for NiTi.

Per cycle (strain %, stress MPa):

* loading branch = cycle start → maximum strain; unloading branch = maximum strain → end of
  unloading (first σ = 0 crossing, else the minimum stress).
* residual strain = strain at the end of unloading (absolute, i.e. accumulated since the test
  start); ``cycle_residual_pct`` = residual − strain at the start of the cycle.
* W_load = ∫σ dε along loading, W_rec = ∫σ dε recovered on unloading, dissipated energy
  (hysteresis loop area, incl. any unrecovered strain) = W_load − W_rec, all in MJ/m³
  (ε in mm/mm). Efficiency η = W_rec / W_load.
* plateau stresses σ_ms (forward, loading) and σ_as (reverse, unloading) are averaged in a small
  strain window around the mid-plateau strain ε_p on each branch; Δσ = σ_ms − σ_as.
  ε_p = ε_lo + f·(ε_max − ε_lo) with ε_lo = max(cycle start, residual) (``plateau_ref="range"``,
  robust to accumulated residual strain) or ε_p = f·ε_max (``plateau_ref="absolute"``).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, signal

from .analysis_common import smooth, trapz
from .dataset import Dataset

__all__ = [
    "Cycle", "CycleMetrics", "split_cycles", "cycle_metrics", "fatigue_summary",
    "fatigue_overview", "fatigue_dataset", "cycle_datasets", "FATIGUE_COLUMNS",
    "normalise_signs", "analyse_cyclic", "CyclicResult", "cycle_warnings", "SaturationFit",
    "fit_saturating", "bare_column_name", "FIT_THRESHOLDS",
]


def normalise_signs(strain, stress, compression="auto") -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Return (strain, stress, strain_flipped, stress_flipped) with positive magnitudes.

    compression: "auto" (flip a channel whose largest excursion is negative — compression recorded
    as negative by the machine), "yes"/True (invert both signs), "no"/False (leave as recorded).
    """
    e = np.asarray(strain, float).ravel()
    s = np.asarray(stress, float).ravel()
    mode = {True: "yes", False: "no", None: "auto"}.get(compression, str(compression).lower())
    if mode == "no":
        return e, s, False, False
    if mode == "yes":
        return -e, -s, True, True
    fe = bool(np.isfinite(e).any() and np.nanmax(e) < -np.nanmin(e))
    fs = bool(np.isfinite(s).any() and np.nanmax(s) < -np.nanmin(s))
    return (-e if fe else e), (-s if fs else s), fe, fs


@dataclass
class Cycle:
    number: int
    indices: np.ndarray        # indices into the original arrays (consecutive cycles share the turning point)


def split_cycles(strain, stress=None, cycle=None, time=None, prominence: float | None = None) -> list[Cycle]:
    """Split a cyclic test into cycles.

    If ``cycle`` (cycle-number column) is given, contiguous runs of equal cycle number are used.
    Otherwise load reversals are detected as strain maxima (``scipy.signal.find_peaks`` with
    ``prominence`` in % strain, default 30 % of the strain range); cycles are separated at the
    strain minimum between consecutive maxima. ``time`` (optional) only fixes the point order.
    """
    e = np.asarray(strain, float).ravel()
    n = e.size
    order = np.argsort(np.asarray(time, float), kind="stable") if time is not None else np.arange(n)
    if n == 0:
        return []
    if cycle is not None:
        c = np.asarray(cycle, float).ravel()[order]
        out: list[Cycle] = []
        i = 0
        while i < n:
            j = i
            while j < n and (c[j] == c[i] or (np.isnan(c[j]) and np.isnan(c[i]))):
                j += 1
            if np.isfinite(c[i]):
                out.append(Cycle(int(round(c[i])), order[i:j]))
            i = j
        return out
    eo = e[order]
    good = np.isfinite(eo)
    if good.sum() < 3:
        return [Cycle(1, order)]
    ef = np.interp(np.arange(n), np.flatnonzero(good), eo[good])
    es = smooth(ef, max(3, min(11, n // 50 * 2 + 3)), polyorder=1)
    prom = prominence if prominence else 0.3 * float(np.ptp(es))
    if prom <= 0:
        return [Cycle(1, order)]
    pk, _ = signal.find_peaks(np.r_[es, -np.inf], prominence=prom)
    pk = pk[pk < n]
    if pk.size == 0:
        return [Cycle(1, order)]
    bounds = [0]
    for p1, p2 in zip(pk[:-1], pk[1:]):
        bounds.append(int(p1 + np.argmin(ef[p1:p2 + 1])))
    bounds.append(n - 1)
    return [Cycle(k + 1, order[bounds[k]:bounds[k + 1] + 1]) for k in range(len(bounds) - 1)]


@dataclass
class CycleMetrics:
    cycle: int
    n_points: int
    start_strain_pct: float
    max_strain_pct: float
    max_stress_MPa: float
    residual_strain_pct: float
    cycle_residual_pct: float
    recoverable_strain_pct: float
    loading_work_MJ_m3: float
    recovered_work_MJ_m3: float
    dissipated_energy_MJ_m3: float
    efficiency: float
    plateau_strain_pct: float
    sigma_ms_MPa: float
    sigma_as_MPa: float
    delta_sigma_MPa: float
    i_max: int = 0             # index (original arrays) of the max-strain point
    i_end: int = 0             # index of the end-of-unloading point
    complete: bool = True      # False: no unloading back to ≤ end_frac·σ_max (loading-only / cut-off cycle)


def _branch_stress_at(eb, sb, ep, hw):
    m = np.abs(eb - ep) <= hw
    if m.sum() >= 2:
        return float(np.mean(sb[m]))
    # interpolate at the first crossing
    d = eb - ep
    cross = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
    if cross.size:
        i = cross[0]
        t = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0.0
        return float(sb[i] + t * (sb[i + 1] - sb[i]))
    return float("nan")


def cycle_metrics(strain_pct, stress_MPa, cycles: list[Cycle] | None = None, *, cycle=None, time=None,
                  plateau_frac: float = 0.5, plateau_window_frac: float = 0.04,
                  plateau_ref: str = "range", prominence: float | None = None,
                  compression="auto", end_frac: float = 0.10) -> list[CycleMetrics]:
    """Per-cycle superelastic metrics (see module docstring for definitions).

    compression: sign handling, see :func:`normalise_signs` (default "auto": compression-signed data
    — negative strain/stress — is analysed as positive magnitudes).
    end_frac: a cycle is *complete* only if unloading brings the stress back to ≤ ``end_frac`` × its
    maximum; incomplete cycles (loading only, test stopped) get NaN for residual strain, recovered
    and dissipated energy, σ_as and Δσ, and ``complete=False``.
    """
    e_all, s_all, _fe, _fs = normalise_signs(strain_pct, stress_MPa, compression)
    if cycles is None:
        cycles = split_cycles(e_all, s_all, cycle=cycle, time=time, prominence=prominence)
    out: list[CycleMetrics] = []
    for cy in cycles:
        idx = cy.indices[np.isfinite(e_all[cy.indices]) & np.isfinite(s_all[cy.indices])]
        if idx.size < 5:
            continue
        e, s = e_all[idx], s_all[idx]
        ipk = int(np.argmax(e))
        eL, sL = e[: ipk + 1], s[: ipk + 1]
        eU, sU = e[ipk:], s[ipk:]
        # end of unloading
        neg = np.flatnonzero(sU <= 0)
        if neg.size and neg[0] > 0:
            k = int(neg[0])
            t = sU[k - 1] / (sU[k - 1] - sU[k]) if sU[k - 1] != sU[k] else 0.0
            res = float(eU[k - 1] + t * (eU[k] - eU[k - 1]))
            eUi = np.r_[eU[:k], res]
            sUi = np.r_[sU[:k], 0.0]
            j_end = k
        else:
            k = int(np.argmin(sU))
            res = float(eU[k])
            eUi, sUi = eU[: k + 1], sU[: k + 1]
            j_end = k
        smax_c = float(np.max(s))
        complete = bool(eU.size >= 3 and smax_c > 0 and float(np.min(sU)) <= end_frac * smax_c)
        W_load = trapz(sL, eL / 100.0)
        W_rec = -trapz(sUi, eUi / 100.0)
        e0 = float(e[0])
        emax = float(e[ipk])
        if plateau_ref == "absolute":
            ep = plateau_frac * emax
            rng = emax
        else:
            lo = max(e0, res)
            ep = lo + plateau_frac * (emax - lo)
            rng = emax - lo
        hw = 0.5 * plateau_window_frac * rng
        sms = _branch_stress_at(eL, sL, ep, hw)
        sas = _branch_stress_at(eUi, sUi, ep, hw)
        if not complete:
            nan = float("nan")
            res, W_rec, sas = nan, nan, nan
        out.append(CycleMetrics(
            cycle=cy.number, n_points=int(idx.size), start_strain_pct=e0, max_strain_pct=emax,
            max_stress_MPa=float(np.max(s)), residual_strain_pct=res, cycle_residual_pct=res - e0,
            recoverable_strain_pct=emax - res, loading_work_MJ_m3=W_load, recovered_work_MJ_m3=W_rec,
            dissipated_energy_MJ_m3=W_load - W_rec,
            efficiency=W_rec / W_load if W_load > 0 else float("nan"),
            plateau_strain_pct=float(ep), sigma_ms_MPa=sms, sigma_as_MPa=sas, delta_sigma_MPa=sms - sas,
            i_max=int(idx[ipk]), i_end=int(idx[min(ipk + j_end, idx.size - 1)]), complete=complete))
    return out


def cycle_warnings(metrics: list[CycleMetrics]) -> list[str]:
    """Warnings for incomplete cycles (excluded from dissipated-energy statistics)."""
    bad = [m.cycle for m in metrics if not m.complete]
    if not bad:
        return []
    lst = ", ".join(map(str, bad[:10])) + (" …" if len(bad) > 10 else "")
    return [f"{len(bad)} incomplete cycle(s) (no unloading back to ≤ 10 % of σ_max: loading-only or cut-off "
            f"data): cycle {lst}. Their residual strain, σ_as, Δσ and dissipated energy are not reported and "
            "they are excluded from the fatigue statistics."]


@dataclass
class CyclicResult:
    """Result of :func:`analyse_cyclic` (metrics + the sign-normalised data that was analysed)."""
    metrics: list[CycleMetrics]
    cycles: list[Cycle]
    strain_pct: np.ndarray = field(repr=False)
    stress_MPa: np.ndarray = field(repr=False)
    strain_sign: float = 1.0          # multiply the analysed values by these to get the recorded ones
    stress_sign: float = 1.0
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def complete_metrics(self) -> list[CycleMetrics]:
        return [m for m in self.metrics if m.complete]


def analyse_cyclic(strain_pct, stress_MPa, *, cycle=None, time=None, compression="auto",
                   prominence: float | None = None, plateau_frac: float = 0.5,
                   plateau_window_frac: float = 0.04, plateau_ref: str = "range",
                   end_frac: float = 0.10) -> CyclicResult:
    """Sign normalisation + cycle splitting + per-cycle metrics + warnings, in one call."""
    e, s, fe, fs = normalise_signs(strain_pct, stress_MPa, compression)
    if np.isfinite(e).sum() < 5 or np.isfinite(s).sum() < 5:
        raise ValueError("Not enough valid strain/stress points for a cyclic analysis")
    warnings: list[str] = []
    if fe or fs:
        which = " and ".join(n for n, f in (("strain", fe), ("stress", fs)) if f)
        warnings.append(f"Compression-signed data: {which} sign inverted so that loops, plateaus and areas are "
                        "computed on positive magnitudes.")
    cycles = split_cycles(e, s, cycle=cycle, time=time, prominence=prominence)
    metrics = cycle_metrics(e, s, cycles, plateau_frac=plateau_frac, plateau_window_frac=plateau_window_frac,
                            plateau_ref=plateau_ref, compression="no", end_frac=end_frac)
    warnings += cycle_warnings(metrics)
    if metrics and all(not m.complete for m in metrics):
        warnings.append("No complete cycle: dissipated energy cannot be computed.")
    params = {"compression": str(compression), "strain_flipped": fe, "stress_flipped": fs,
              "prominence_pct": prominence, "cycle_column": cycle is not None, "plateau_frac": plateau_frac,
              "plateau_window_frac": plateau_window_frac, "plateau_ref": plateau_ref, "end_frac": end_frac,
              "n_cycles": len(metrics), "n_incomplete": sum(not m.complete for m in metrics)}
    return CyclicResult(metrics, cycles, e, s, -1.0 if fe else 1.0, -1.0 if fs else 1.0, warnings, [], params)


# column header -> (CycleMetrics attribute | callable, dimension)
FATIGUE_COLUMNS: dict[str, tuple[str, str]] = {
    "Cycle": ("cycle", "count"),
    "Max strain (%)": ("max_strain_pct", "strain"),
    "Max stress (MPa)": ("max_stress_MPa", "stress"),
    "Residual strain (%)": ("residual_strain_pct", "strain"),
    "Accumulated residual strain (%)": ("_accum", "strain"),
    "Cycle residual strain (%)": ("cycle_residual_pct", "strain"),
    "Recoverable strain (%)": ("recoverable_strain_pct", "strain"),
    "Upper plateau σ_ms (MPa)": ("sigma_ms_MPa", "stress"),
    "Lower plateau σ_as (MPa)": ("sigma_as_MPa", "stress"),
    "Stress hysteresis Δσ (MPa)": ("delta_sigma_MPa", "stress"),
    "Dissipated energy (MJ/m³)": ("dissipated_energy_MJ_m3", "energy_density"),
    "Loading work (MJ/m³)": ("loading_work_MJ_m3", "energy_density"),
    "Recovered work (MJ/m³)": ("recovered_work_MJ_m3", "energy_density"),
    "Efficiency (-)": ("efficiency", "dimensionless"),
}


def fatigue_summary(metrics: list[CycleMetrics]) -> pd.DataFrame:
    """Table of the functional-fatigue metrics vs cycle number (unit-bearing headers).

    'Accumulated residual strain' is the residual strain relative to the start of the first cycle.
    """
    if not metrics:
        return pd.DataFrame(columns=list(FATIGUE_COLUMNS))
    e_start = metrics[0].start_strain_pct
    data = {}
    for col, (attr, _dim) in FATIGUE_COLUMNS.items():
        if attr == "_accum":
            data[col] = [m.residual_strain_pct - e_start for m in metrics]
        else:
            data[col] = [getattr(m, attr) for m in metrics]
    return pd.DataFrame(data)


def fatigue_overview(metrics: list[CycleMetrics]) -> list[tuple[str, float, str]]:
    """First-vs-last summary rows (quantity, value, unit) for the functional-fatigue test."""
    if not metrics:
        return []
    allm = metrics
    metrics = [m for m in metrics if m.complete] or metrics
    a, b = metrics[0], metrics[-1]

    def pct(x0, x1):
        return 100.0 * (x1 - x0) / x0 if x0 else float("nan")

    rows = [
        ("Cycles analysed", len(allm), "#"),
        ("Residual strain, first cycle", a.residual_strain_pct - a.start_strain_pct, "%"),
        ("Accumulated residual strain, last cycle", b.residual_strain_pct - a.start_strain_pct, "%"),
        ("σ_ms first cycle", a.sigma_ms_MPa, "MPa"), ("σ_ms last cycle", b.sigma_ms_MPa, "MPa"),
        ("σ_ms change", pct(a.sigma_ms_MPa, b.sigma_ms_MPa), "%"),
        ("σ_as first cycle", a.sigma_as_MPa, "MPa"), ("σ_as last cycle", b.sigma_as_MPa, "MPa"),
        ("Δσ first cycle", a.delta_sigma_MPa, "MPa"), ("Δσ last cycle", b.delta_sigma_MPa, "MPa"),
        ("Dissipated energy first cycle", a.dissipated_energy_MJ_m3, "MJ/m³"),
        ("Dissipated energy last cycle", b.dissipated_energy_MJ_m3, "MJ/m³"),
        ("Dissipated energy change", pct(a.dissipated_energy_MJ_m3, b.dissipated_energy_MJ_m3), "%"),
        ("Recoverable strain last cycle", b.recoverable_strain_pct, "%"),
    ]
    # stabilisation: first cycle after which the per-cycle residual increment stays < 0.01 %
    inc = np.diff([m.residual_strain_pct for m in metrics])
    stab = float("nan")
    for k in range(inc.size):
        if np.all(np.abs(inc[k:]) < 0.01):
            stab = float(metrics[k + 1].cycle)
            break
    rows.append(("Stabilisation cycle (Δε_res < 0.01 %/cycle)", int(stab) if np.isfinite(stab) else stab, "#"))
    n_inc = len(allm) - len([m for m in allm if m.complete])
    if n_inc:
        rows.insert(1, ("Incomplete cycles (excluded)", n_inc, "#"))
    return rows


_DIM_UNIT = {"count": "#", "strain": "%", "stress": "MPa", "energy_density": "MJ/m³", "dimensionless": ""}


def bare_column_name(col: str) -> str:
    """'Dissipated energy (MJ/m³)' → 'Dissipated energy' (the unit lives in the channel, so axis labels
    are not doubled)."""
    return re.sub(r"\s*\([^()]*\)\s*$", "", col).strip() or col


def fatigue_dataset(metrics: list[CycleMetrics], name: str = "Fatigue summary",
                    source: Dataset | None = None) -> Dataset:
    """Derived Dataset (cycle number vs metrics) for plotting functional fatigue.

    Column names carry no unit suffix (e.g. 'Dissipated energy'); the unit is in the channel so
    axis labels read 'Dissipated energy (MJ/m³)' exactly once. Use :func:`bare_column_name` to map a
    :data:`FATIGUE_COLUMNS` header to its column."""
    df = fatigue_summary(metrics)
    cols = {bare_column_name(c): (df[c].to_numpy(float), FATIGUE_COLUMNS[c][1], _DIM_UNIT[FATIGUE_COLUMNS[c][1]])
            for c in df.columns}
    ds = Dataset.from_arrays(name, cols, kind="generic", default_x="Cycle", default_y="Residual strain")
    if source is not None:
        ds.source_path = source.source_path
        ds.metadata = dict(source.metadata)
        ds.metadata["derived_from"] = source.name
    return ds


def cycle_datasets(ds: Dataset, cycles: list[Cycle], every: int = 1) -> list[Dataset]:
    """One sub-Dataset per cycle (every ``every``-th cycle, plus the last), kind 'cyclic'."""
    out = []
    n = len(ds.data)
    for k, cy in enumerate(cycles):
        if every > 1 and k % every != 0 and k != len(cycles) - 1:
            continue
        mask = np.zeros(n, bool)
        mask[cy.indices] = True
        sub = ds.subset(mask, name=f"{ds.name} – cycle {cy.number}")
        sub.metadata["cycle"] = cy.number
        out.append(sub)
    return out


# --------------------------------------------------------------------------- fatigue trend fits
FIT_THRESHOLDS = {"strain": 0.01, "stress": 0.5, "energy_density": 0.05}   # change per cycle


@dataclass
class SaturationFit:
    """y(N) = a + b·(1 − exp(−(N − 1)/τ)); ``saturation`` = a + b."""
    a: float
    b: float
    tau: float
    saturation: float
    r2: float
    stabilisation_cycle: float      # first N with |dy/dN| < threshold (nan if never within the data)
    threshold: float
    n: int
    x_fit: np.ndarray = field(repr=False, default=None)
    y_fit: np.ndarray = field(repr=False, default=None)

    def __call__(self, N):
        N = np.asarray(N, float)
        return self.a + self.b * (1.0 - np.exp(-(N - 1.0) / self.tau))


def fit_saturating(N, y, threshold: float | None = None) -> SaturationFit:
    """Least-squares saturating-exponential fit of a functional-fatigue metric vs cycle number.

    threshold: per-cycle change defining the stabilisation cycle (default 1 % of |b| per cycle).
    The stabilisation cycle is the first data cycle N with |dy/dN| = |b|/τ·e^(−(N−1)/τ) < threshold.
    """
    N = np.asarray(N, float).ravel()
    y = np.asarray(y, float).ravel()
    m = np.isfinite(N) & np.isfinite(y)
    N, y = N[m], y[m]
    if N.size < 4:
        raise ValueError("Need at least 4 cycles for a saturation fit")
    rngN = float(np.ptp(N)) or 1.0

    def f(x, a, b, tau):
        return a + b * (1.0 - np.exp(-(x - 1.0) / tau))

    best = None
    for tau0 in (rngN / 20, rngN / 5, rngN / 2, rngN):
        p0 = [y[0], y[-1] - y[0], max(tau0, 0.05)]
        try:
            p, _ = optimize.curve_fit(f, N, y, p0=p0, bounds=([-np.inf, -np.inf, 1e-3], [np.inf, np.inf, 1e4 * rngN]),
                                      maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((f(N, *p) - y) ** 2))
        if best is None or sse < best[1]:
            best = (p, sse)
    if best is None:
        raise ValueError("Saturation fit did not converge")
    (a, b, tau), sse = best
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    thr = float(threshold) if threshold else 0.01 * abs(b)
    stab = float("nan")
    for n_ in np.unique(N):
        if abs(b) / tau * np.exp(-(n_ - 1.0) / tau) < thr:
            stab = float(n_)
            break
    xf = np.linspace(N.min(), N.max(), 200)
    return SaturationFit(float(a), float(b), float(tau), float(a + b), float(r2), stab, thr, int(N.size), xf,
                         f(xf, a, b, tau))


def metrics_as_dicts(metrics: list[CycleMetrics]) -> list[dict]:
    return [asdict(m) for m in metrics]
