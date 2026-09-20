"""Calibration: the part of the product the build plan says never to cut."""

from .conformal import ConformalPredictor, PredictionSet, fit_conformal
from .metrics import (
    CalibrationReport,
    ReliabilityBin,
    adaptive_calibration_error,
    brier,
    coverage,
    expected_calibration_error,
    maximum_calibration_error,
    mean_set_size,
    negative_log_likelihood,
    reliability_bins,
    report,
)
from .temperature import TemperatureScaler, fit_temperature

__all__ = [
    "CalibrationReport",
    "ConformalPredictor",
    "PredictionSet",
    "ReliabilityBin",
    "TemperatureScaler",
    "adaptive_calibration_error",
    "brier",
    "coverage",
    "expected_calibration_error",
    "fit_conformal",
    "fit_temperature",
    "maximum_calibration_error",
    "mean_set_size",
    "negative_log_likelihood",
    "reliability_bins",
    "report",
]
