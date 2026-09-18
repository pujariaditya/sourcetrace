"""Trainable model components for the source-tracing method.

Reconstructed from the published method description and the ``method.py``
interface contract — this is NOT recovered champion code and does NOT reproduce the
reported FPR95 bit-for-bit; see ``head.py`` for the full RECONSTRUCTION NOTICE and for
the two properties that are easy to misread: the frozen (never-optimized) codec
projection, and the fact that ``RawWhitenTransfer``'s four blocks are feature-axis
quarters rather than WavLM layers.
"""
from __future__ import annotations

from .head import FactorizedGatedHead, RawWhitenTransfer, build_head_pair

__all__ = ["FactorizedGatedHead", "RawWhitenTransfer", "build_head_pair"]
