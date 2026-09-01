"""doc-classifier: classify business documents by type, with calibrated confidence."""

from __future__ import annotations

__version__ = "0.1.0"

from .types import Document, Features, Prediction

__all__ = ["Document", "Features", "Prediction", "__version__"]
