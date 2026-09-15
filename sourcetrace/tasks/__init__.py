"""Standalone evaluation protocols for the two source-tracing benchmarks.

* :mod:`sourcetrace.tasks.mlaad_v5`  -- MLAAD v5 PANDA open-set evaluation.
* :mod:`sourcetrace.tasks.stopa`     -- STOPA verification (6 EERs, unknown-attack).
* :mod:`sourcetrace.tasks._features` -- private per-group cached-feature loader
  shared by both protocols.

The metric formulas they score with are one level up, in
:mod:`sourcetrace.metrics`, and are re-exported here so a caller that wants a
protocol and its metric needs one import.

Every protocol is a plain function: it takes the method object + cached features,
computes the published metrics with numpy/sklearn, and returns a result dataclass
(``.as_dict()`` for JSON). No subprocess workers, no GPU file-locks, no composite
scoring.

Each protocol exposes a ``fit_*`` (train only), an ``eval_*`` (inference on a
pre-fit method) and a combined ``evaluate_*`` entry point.

.. warning::

   :func:`ood_eer` and :func:`stopa_eer` implement two DIFFERENT, deliberately
   non-interchangeable EER conventions -- asymmetric ``fpr[eer_index]`` for MLAAD
   and symmetric ``(fpr+fnr)/2`` for STOPA. See
   :mod:`sourcetrace.metrics.protocol`. Do not unify them.
"""
from __future__ import annotations

from ..metrics.protocol import (
    fpr95_panda,
    ood_eer,
    stopa_eer,
)
from .mlaad_v5 import (
    MlaadV5Result,
    eval_mlaad_v5,
    evaluate_mlaad_v5,
    fit_mlaad_v5,
)
from .stopa import (
    StopaResult,
    eval_stopa,
    evaluate_stopa,
    fit_stopa,
)

__all__ = [
    "fpr95_panda", "ood_eer", "stopa_eer",
    "MlaadV5Result", "fit_mlaad_v5", "eval_mlaad_v5", "evaluate_mlaad_v5",
    "StopaResult", "fit_stopa", "eval_stopa", "evaluate_stopa",
]
