"""
Correlation module for extracting cross-asset signals.
"""
from .cross_asset import inject_cross_asset_features
from .matrix_builder import CorrelationMatrixBuilder

__all__ = ["CorrelationMatrixBuilder", "inject_cross_asset_features"]
