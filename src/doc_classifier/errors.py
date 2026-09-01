"""Errors the CLI reports as exit code 1."""

from __future__ import annotations


class DocClassifierError(Exception):
    """Base class for user-facing input/usage failures."""


class DataFormatError(DocClassifierError):
    """Training/evaluation input could not be read as labelled documents."""


class ModelError(DocClassifierError):
    """A model file could not be read, or is unusable for the request."""
