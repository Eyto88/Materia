"""LabStudio analysis API (pure Python, no Qt).

Re-exports the general helpers and the DSC, mechanical and cyclic analyses:

General   : :func:`find_peaks_xy`, :func:`integrate_xy`, :func:`linear_fit`
DSC       : :func:`split_dsc_segments`, :func:`analyse_dsc_segment`, :func:`analyse_dsc`,
            :func:`delta_T_ad_estimate`, :class:`DSCPeak`, :class:`DSCAnalysis`
Mechanical: :func:`stress_strain`, :func:`tensile_properties`, :func:`compression_properties`,
            :class:`MechanicalResult`
Cyclic    : :func:`split_cycles`, :func:`cycle_metrics`, :func:`analyse_cyclic`, :func:`fatigue_summary`,
            :func:`fatigue_overview`, :func:`fatigue_dataset`, :func:`cycle_datasets`, :func:`fit_saturating`
See also :mod:`labstudio.core.batch` (batch analysis) and :mod:`labstudio.core.elastocaloric`
(Clausius–Clapeyron, measured ΔT / COP).

All values are in LabStudio canonical units (°C, W/g or mW, J/g, %, MPa, GPa for moduli, MJ/m³).
"""
from __future__ import annotations

from .analysis_common import (Integration, LinearFit, XYPeak, find_peaks_xy, integrate_xy,
                              linear_fit, smooth, trapz)
from .analysis_cyclic import (FATIGUE_COLUMNS, FIT_THRESHOLDS, Cycle, CycleMetrics, CyclicResult, SaturationFit,
                              analyse_cyclic, bare_column_name, cycle_datasets, cycle_metrics, cycle_warnings,
                              fatigue_dataset, fatigue_overview, fatigue_summary, fit_saturating, normalise_signs,
                              split_cycles)
from .analysis_dsc import (DEFAULT_RATE_K_PER_MIN, DSCAnalysis, DSCPeak, DSCSegmentResult, DSCSegments, PeakList,
                           analyse_dsc, analyse_dsc_segment, delta_T_ad_estimate, detect_endo_up,
                           dsc_segment_datasets, estimate_noise, name_sma_peaks, segment_labels, select_segments,
                           split_dsc_segments)
from .analysis_mech import (CompressionResult, MechanicalResult, TensileResult, auto_modulus_fit,
                            compliance_correct, compression_properties, estimate_compliance,
                            mechanical_sanity_warnings, stress_strain, tensile_properties)

__all__ = [
    "Integration", "LinearFit", "XYPeak", "find_peaks_xy", "integrate_xy", "linear_fit", "smooth", "trapz",
    "FATIGUE_COLUMNS", "Cycle", "CycleMetrics", "cycle_datasets", "cycle_metrics", "fatigue_dataset",
    "fatigue_overview", "fatigue_summary", "split_cycles", "FIT_THRESHOLDS", "CyclicResult", "SaturationFit",
    "analyse_cyclic", "bare_column_name", "cycle_warnings", "fit_saturating", "normalise_signs",
    "DEFAULT_RATE_K_PER_MIN", "PeakList", "dsc_segment_datasets", "estimate_noise", "segment_labels",
    "select_segments", "compliance_correct", "estimate_compliance", "mechanical_sanity_warnings",
    "DSCAnalysis", "DSCPeak", "DSCSegmentResult", "DSCSegments", "analyse_dsc", "analyse_dsc_segment",
    "delta_T_ad_estimate", "detect_endo_up", "name_sma_peaks", "split_dsc_segments",
    "CompressionResult", "MechanicalResult", "TensileResult", "auto_modulus_fit",
    "compression_properties", "stress_strain", "tensile_properties",
]
