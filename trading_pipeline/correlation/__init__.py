"""
Correlation module for extracting cross-asset signals.
"""
from .matrix_builder import CorrelationMatrixBuilder
from .cross_asset import inject_cross_asset_features

__all__ = ["CorrelationMatrixBuilder", "inject_cross_asset_features"]
