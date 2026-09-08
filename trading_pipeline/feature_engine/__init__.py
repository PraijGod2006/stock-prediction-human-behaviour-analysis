"""
Feature engineering module for the trading pipeline.
Exports the necessary classes and functions for building features.
"""

from .aggregator import Aggregator
from .indicators import TechnicalIndicators
from .pipeline import build_features_1min, build_features_5min
from .zscore import ZScoreCalculator

__all__ = [
    'Aggregator',
    'TechnicalIndicators',
    'ZScoreCalculator',
    'build_features_1min',
    'build_features_5min'
]
