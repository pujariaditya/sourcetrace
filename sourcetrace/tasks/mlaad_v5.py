"""MLAAD v5 PANDA open-set source-tracing evaluation (standalone).

Faithful port of the metric flow in the original research grader
(``mlaad_klein.compute_all_metrics`` L838-1000, PRIMARY method-score path
L919-959; the ``train_and_score`` orchestration L738-818), with all
Research-grader coupling removed: no subprocess ``embed_worker`` /
``mlaad_klein_worker``, no GPU file-locks, no outlier-exposure/augmentation
side-channels, no composite scoring. The split, label space, seed and metric
formulas are unchanged, so results reproduce.

Pipeline
--------
1. Build the PANDA family-level split (:func:`sourcetrace.datasets.mlaad.build_split`).
2. Load row-aligned cached features per split
   (:func:`sourcetrace.tasks._features.stack_features`).
3. ``method.fit(train_features, gen_labels=class_id, lang_labels=lang_id, seed)``
   on the ID-only train rows.
4. ``method.score_openset(enrol_banks, eval_X)`` on dev + eval -> a per-sample
   score where *higher = more ID/known* (the method contract). The OOD-positive
   score is ``-score`` (higher = more OOD), exactly as the original negates the
   method output before the repo metrics (L921-925, L932).
5. Metrics (see "Which scores feed which metric" below).

Which scores feed which metric
------------------------------
The three headline numbers deliberately read from *different* score pools:

* ``fpr95`` (:func:`sourcetrace.metrics.protocol.fpr95_panda`, original L936-938 --
  the paper's 3.36 metric) is **calibrated on the DEV-OOD scores and tested on
  the EVAL-ID scores**. Dev-ID scores and eval-OOD scores never enter it. Two of
  the four pools are unused by design; that split is what keeps the threshold
  free of eval information.
* ``ood_eer`` (:func:`sourcetrace.metrics.protocol.ood_eer`, original ``ms_eer_eval``
  L933/L941) uses the **full eval split** -- every eval row, ID and OOD alike,
  contributes a ``(label=1 if OOD, score=-method_score)`` pair. Dev is not
  involved.
* ``id_acc`` uses the **ID eval rows only**, scored against train-derived class
  prototypes.

.. note::

   **The one deliberate protocol deviation is** ``id_acc``. The original
   measured 24-way *softmax-head* accuracy; this module measures top-1 accuracy
   by nearest train-class prototype instead (see :func:`_closed_set_id_acc`).
   Same quantity in intent -- closed-set attribution over the ID classes on the
   ID eval rows -- but a different classifier, so ``id_acc`` is NOT directly
   comparable to a published softmax-head number. ``fpr95`` and ``ood_eer`` are
   byte-identical ports and are comparable.

Leak-safety: ``fit`` sees ONLY train ID features + labels. Dev is used only to
calibrate the OOD threshold; no eval labels enter training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import PROTOCOL
from ..datasets.mlaad import MlaadSplit, MlaadTable, build_split
from ..metrics.protocol import fpr95_panda, ood_eer
from ._features import FeatureSource, stack_features


@dataclass(frozen=True)
class MlaadV5Result:
    """MLAAD v5 evaluation outcome. All three metrics are percentages."""

    fpr95: float           # PANDA FPR@95, percent (compare to SOTA 3.36)
    ood_eer: float         # eval OOD-EER, percent (asymmetric Klein convention)
    id_acc: float          # ID closed-set top-1 accuracy, percent (see module note)
    n_known_classes: int   # number of ID ("known") generator classes
    n_eval: int            # eval rows in total
    n_eval_id: int         # eval rows whose class is known (ID)
    n_eval_ood: int        # eval rows whose class is unseen (OOD)

    def as_dict(self) -> dict[str, float]:
        """The three headline metrics only, for JSON reporting (counts omitted)."""
        return {
            "fpr95": self.fpr95,
            "ood_eer": self.ood_eer,
            "id_acc": self.id_acc,
        }


def _l2(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise along the last axis (``eps``-clipped to survive zero rows).

    Intentionally duplicated in :mod:`sourcetrace.tasks.stopa`: each protocol
    module is meant to be readable and auditable standalone, so they do not
    import numeric helpers from one another.
    """
    return X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), eps, None)


def _enrol_banks(train_emb: np.ndarray, class_id: np.ndarray,
                 n_known: int) -> dict[int, np.ndarray]:
    """Per-ID-class embedding banks (for ``score_openset``'s fallback path).

    Classes with no train rows are omitted rather than mapped to an empty array.
    """
    return {c: train_emb[class_id == c]
            for c in range(n_known) if np.any(class_id == c)}


def _closed_set_id_acc(train_emb: np.ndarray, train_cid: np.ndarray,
                       eval_emb: np.ndarray, eval_cid: np.ndarray,
                       n_known: int) -> float:
    """Top-1 accuracy over ID eval rows via nearest train-class prototype.

    .. warning::

       **This is the one deliberate deviation from the original protocol.** The
       original reported 24-way *softmax-head* accuracy
       (``id_accuracy_report`` L453-471); this is a nearest-prototype stand-in.
       Both measure the same intended quantity -- closed-set attribution over the
       ID classes, evaluated on the ID eval rows -- but through a different
       classifier, so the resulting ``id_acc`` must NOT be quoted as
       reproducing a published softmax-head accuracy. Unlike ``fpr95`` and
       ``ood_eer`` (byte-identical ports), this number is our own construction.

    The stand-in keeps the module dependency-light and leak-safe: prototypes are
    L2-normed class means of the *train* embeddings only, and prediction is the
    argmax cosine against them. Assumes every one of the ``n_known`` classes has
    at least one train row (guaranteed by the PANDA split builder); a genuinely
    empty class would make its prototype ``nan``.

    Returns:
        Top-1 accuracy as a percentage, or ``nan`` if the eval split has no ID
        rows at all.
    """
    id_mask = eval_cid < n_known
    if not np.any(id_mask):
        return float("nan")
    protos = np.stack([_l2(train_emb[train_cid == c].mean(0))
                       for c in range(n_known)])
    pred = np.argmax(_l2(eval_emb[id_mask]) @ protos.T, axis=1)
    return float(np.mean(pred == eval_cid[id_mask]) * 100.0)


def fit_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> Any:
    """Fit a source-tracing method on the MLAAD v5 PANDA *train* split (no eval).

    Train entry point for the split train/inference workflow. Fits ONLY on the
    ID-only train rows (leak-safe: dev/eval never enter ``fit``). Returns the same
    (now-fitted) ``method`` so it can be checkpointed and later handed to
    :func:`eval_mlaad_v5`.

    Args:
        method: a method implementing ``fit(features, gen_labels, lang_labels, seed)``.
        features: a :data:`FeatureSource` resolving each table's cached features.
        split: prebuilt :class:`MlaadSplit`; built via
            :func:`sourcetrace.datasets.mlaad.build_split` when ``None``.
        seed: fit/split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        The fitted ``method``.
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)
    train: MlaadTable = split.train
    train_X = stack_features(train, features)
    method.fit(train_X, train.class_id, train.lang_id, seed)
    return method


def eval_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> MlaadV5Result:
    """Evaluate a PRE-FIT source-tracing method on the MLAAD v5 PANDA protocol.

    Inference entry point: assumes ``method`` is already fitted (via
    :func:`fit_mlaad_v5` or :meth:`sourcetrace.method.Method.load`) and does NOT
    call ``fit``. Computes the identical metrics as :func:`evaluate_mlaad_v5`.

    ``fpr95`` calibrates on DEV-OOD scores and tests on EVAL-ID scores;
    ``ood_eer`` uses the whole eval split; ``id_acc`` uses the ID eval rows. See
    the module docstring for why those pools differ.

    Args:
        method: a fitted method implementing ``embed(X)`` /
            ``score_openset(enrol_by_class, eval_X)`` (higher = more ID).
        features: a :data:`FeatureSource` resolving each table's cached features.
        split: prebuilt :class:`MlaadSplit`; built when ``None`` (same seed).
        seed: split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        An :class:`MlaadV5Result` with ``fpr95`` / ``ood_eer`` / ``id_acc``.
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)

    train: MlaadTable = split.train
    dev: MlaadTable = split.dev
    ev: MlaadTable = split.eval
    n_known = split.n_known_classes

    train_X = stack_features(train, features)
    dev_X = stack_features(dev, features)
    eval_X = stack_features(ev, features)

    train_emb = np.asarray(method.embed(train_X))
    dev_emb = np.asarray(method.embed(dev_X))
    eval_emb = np.asarray(method.embed(eval_X))

    banks = _enrol_banks(train_emb, train.class_id, n_known)

    # method score: higher = more ID/known -> OOD-positive score is its negative.
    dev_ood_score = -np.asarray(method.score_openset(banks, dev_emb), dtype=np.float64)
    eval_ood_score = -np.asarray(method.score_openset(banks, eval_emb), dtype=np.float64)

    dev_is_ood = dev.class_id >= n_known
    eval_is_ood = ev.class_id >= n_known
    eval_is_id = ~eval_is_ood

    # -- FPR95 (PANDA) --------------------------------------------------------- #
    # Calibrate the threshold on DEV-OOD scores, test it on EVAL-ID scores.
    # Dev-ID and eval-OOD scores are deliberately NOT used by this metric.
    fpr95 = fpr95_panda(dev_ood_score[dev_is_ood],
                        eval_ood_score[eval_is_id]) * 100.0

    # -- OOD-EER --------------------------------------------------------------- #
    # Uses the FULL eval split (ID rows and OOD rows together), unlike FPR95.
    # Labels 1=OOD; score higher = more OOD. Asymmetric fpr[eer_index] convention.
    eer_frac, _ = ood_eer(eval_is_ood.astype(np.int64), eval_ood_score)

    # -- Closed-set ID accuracy (ID eval rows only; protocol deviation) -------- #
    id_acc = _closed_set_id_acc(train_emb, train.class_id,
                                eval_emb, ev.class_id, n_known)

    return MlaadV5Result(
        fpr95=fpr95,
        ood_eer=eer_frac * 100.0,
        id_acc=id_acc,
        n_known_classes=n_known,
        n_eval=len(ev),
        n_eval_id=int(eval_is_id.sum()),
        n_eval_ood=int(eval_is_ood.sum()),
    )


def evaluate_mlaad_v5(
    method: Any,
    features: FeatureSource,
    split: MlaadSplit | None = None,
    seed: int | None = None,
) -> MlaadV5Result:
    """Fit + evaluate a source-tracing method on the MLAAD v5 PANDA protocol.

    Back-compatible combined entry point: fits on train then evaluates on dev/eval
    in one call. Equivalent to ``eval_mlaad_v5(fit_mlaad_v5(method, features,
    split, seed), features, split, seed)``. The metric math is unchanged.

    Args:
        method: a method object implementing the contract
            ``fit(features, gen_labels, lang_labels, seed)`` /
            ``embed(X) -> (N, emb_dim)`` (L2-normed) /
            ``score_openset(enrol_by_class, eval_X) -> (N,)`` (higher = more ID).
        features: a :data:`FeatureSource` resolving each table's per-group cached
            ``.npy`` features (see :mod:`sourcetrace.tasks._features`).
        split: a prebuilt :class:`MlaadSplit`; built via
            :func:`sourcetrace.datasets.mlaad.build_split` when ``None``.
        seed: fit/split seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        An :class:`MlaadV5Result` with ``fpr95`` / ``ood_eer`` / ``id_acc`` (all
        percentages).
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    split = split if split is not None else build_split(seed=seed)
    fit_mlaad_v5(method, features, split, seed)
    return eval_mlaad_v5(method, features, split, seed)
