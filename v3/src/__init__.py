# V3 Source Package
"""
Production-ready modules for V3 Climate-Aware Transformer.
"""

from .config import V3Config
from .inference import V3Forecaster
from .model import GatedResidualNetwork, V3ClimateTransformer
from .preprocessing import V3Preprocessor

__version__ = "3.0.0"
__all__ = [
    "V3Config",
    "V3ClimateTransformer",
    "GatedResidualNetwork",
    "V3Preprocessor",
    "V3Forecaster",
]
