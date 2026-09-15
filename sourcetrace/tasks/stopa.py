"""STOPA open-set source-tracing verification evaluation (standalone).

Faithful port of the verification core in the original research grader
(``stopa_verif.py``): ``build_fingerprints`` (L233-262), ``score_trials``
(L265-278), ``compute_six_eers`` (L281-369) and the ``run`` orchestration
(L578-615), with all research-grader coupling removed (no subprocess
``embed_worker`` / ``stopa_verif_worker``, no GPU file-locks, no composite
scoring). The partition, fingerprint construction, trial logic and EER formula
are unchanged, so results reproduce.

Pipeline
--------
1. Build the STOPA tables (:func:`sourcetrace.datasets.stopa.build_tables`).
2. Load row-aligned cached features per split
   (:func:`sourcetrace.tasks._features.stack_features`).
3. ``method.fit(eet_features, gen_labels=attack_id, lang_labels=am_id, seed)`` on
   EET ONLY (leak-safe: TEE/Trials never enter fit).
4. ``method.embed`` the TEE (enroll) and Trials (probe) features.
5. Build ATK / AM / VM fingerprints = L2-normed means of the TEE embeddings
   (per attack, per acoustic-model, per vocoder).
6. Cosine-score every Trials probe against every candidate fingerprint and
   compute the KNOWN- and UNKNOWN-scenario EER for each granularity -> 6 numbers.
   Headline = the UNKNOWN-scenario ATK EER; this repository measures 9.3303
   (``results/stopa_measured.json``, split seed 42 / fit seed 0).

Leak-safety: ``fit`` sees ONLY EET features + labels; fingerprints come from TEE
embeddings, probes from Trials; no eval labels enter training.

NOT A ZERO-SHOT SYSTEM. Step 3 fits on EET, so the headline is a *fitted* result. The
lowest published figure, Chhibber et al.'s 16.43 (Odyssey 2026), is titled zero-shot and
fits on no STOPA split at all; the two are not interchangeable and code or prose that
puts them side by side has to say which is which. STOPA's own trained baselines are the
matched comparison. What is zero-shot here is the *scoring rule* -- an unknown attack is
rejected by cosine distance to enrolled fingerprints, never by a classifier trained on
it -- and that is a different claim from a zero-shot system.

Tested by ``tests/test_stopa_protocol.py``: split disjointness, partition counts, and the
fingerprint -> score -> six-EER path. That file exists because this module calls itself a
faithful port of code that is no longer in the checkout, and agreement with a lost
reference would evidence transcription rather than protocol correctness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import numpy as np

from ..config import PROTOCOL
from ..datasets.stopa import KNOWN_ATTACKS, StopaTables, build_tables
from ..metrics.protocol import stopa_eer
from ._features import FeatureSource, stack_features

_GRANULARITIES = ("ATK", "AM", "VM")


@dataclass(frozen=True)
class StopaResult:
    """STOPA verification outcome. All EERs are percentages.

    .. warning::

       ``n_tee`` and ``n_trials`` are **utterance counts, not trial-pair counts.**
       ``n_trials`` is ``len(trials_emb)`` -- the number of PROBE waveforms in the
       Trials partition (verified: 629,800 wavs = 10 attacks x 62,980). Each probe
       is scored against *every* candidate fingerprint at each granularity, so the
       score matrices that actually feed the EERs are far larger: roughly
       ``n_trials x n_fingerprints[G]`` entries per granularity. Do not report
       ``n_trials`` as "number of trials" in the verification-protocol sense.
    """

    unknown_atk_eer: float                 # headline; see the module note on settings
    six_eers: dict[str, dict[str, float]]  # {G: {"known": eer, "unknown": eer}}
    n_tee: int                             # enrollment utterance count
    n_trials: int                          # PROBE utterance count (not pairs)
    n_fingerprints: dict[str, int] = field(default_factory=dict)  # {G: n_claims}

    def as_dict(self) -> dict[str, Any]:
        """Headline EER + the 6-EER table only, for JSON reporting."""
        return {"unknown_atk_eer": self.unknown_atk_eer, "six_eers": self.six_eers}


def _l2(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalise along the last axis (``eps``-clipped to survive zero rows).

    Intentionally duplicated in :mod:`sourcetrace.tasks.mlaad_v5`: each protocol
    module is meant to be readable and auditable standalone, so they do not
    import numeric helpers from one another.
    """
    return X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), eps, None)


def build_fingerprints(
    tee_emb: np.ndarray,
    tee_attack: list[str],
    attacks_meta: dict[str, dict[str, str]],
) -> dict[str, dict[str, np.ndarray]]:
    """ATK / AM / VM fingerprints from TEE (enroll) embeddings.

    Faithful port of ``stopa_verif.build_fingerprints`` (L233-262). A fingerprint
    is the L2-normed *mean* of the matching TEE embeddings: per attack (ATK), per
    acoustic model (AM), per vocoder (VM). Pooling the actual embeddings makes the
    AM/VM means exact even for unbalanced attack counts.

    Args:
        tee_emb: ``(n_tee, emb_dim)`` TEE enrollment embeddings.
        tee_attack: per-row attack label, row-aligned with ``tee_emb``.
        attacks_meta: ``{attack: {"am": ..., "vm": ...}}`` component metadata.

    Returns:
        ``{G: {label: fingerprint_vec}}`` for ``G`` in ``ATK``/``AM``/``VM``, with
        L2-normed vectors so a dot product equals cosine similarity. Labels within
        each granularity are the sorted unique values.
    """
    tee_emb = np.asarray(tee_emb, dtype=np.float64)
    tee_am = [attacks_meta[a]["am"] for a in tee_attack]
    tee_vm = [attacks_meta[a]["vm"] for a in tee_attack]

    def _means(group_labels: list[str]) -> dict[str, np.ndarray]:
        """L2-normed mean embedding per distinct label, in sorted label order."""
        labels = np.asarray(group_labels)
        return {g: _l2(tee_emb[labels == g].mean(axis=0))
                for g in sorted(set(group_labels))}

    return {"ATK": _means(tee_attack), "AM": _means(tee_am), "VM": _means(tee_vm)}


def score_trials(
    trials_emb: np.ndarray,
    fingerprints: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, Any]]:
    """Cosine-score every probe against every candidate fingerprint per granularity.

    Faithful port of ``stopa_verif.score_trials`` (L265-278). Probe embeddings are
    L2-normed so the dot product is cosine (fingerprints already normed).

    Args:
        trials_emb: ``(n_probe, emb_dim)`` probe embeddings.
        fingerprints: output of :func:`build_fingerprints`.

    Returns:
        ``{G: {"claims": [label, ...], "scores": ndarray[n_probe, n_claim]}}``
        with ``claims`` in sorted label order and ``scores`` column-aligned to it.
    """
    E = _l2(np.asarray(trials_emb, dtype=np.float64))
    out: dict[str, dict[str, Any]] = {}
    for G, fps in fingerprints.items():
        claims = sorted(fps.keys())
        F = np.stack([fps[c] for c in claims], axis=0)
        out[G] = {"claims": claims, "scores": E @ F.T}
    return out


def compute_six_eers(
    trials_emb: np.ndarray,
    trials_attack: list[str],
    attacks_meta: dict[str, dict[str, str]],
    fingerprints: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """The 6 verification EERs (ATK/AM/VM x known/unknown), as percentages.

    Faithful port of ``stopa_verif.compute_six_eers`` (L281-369). For each probe
    and each candidate claim ``c`` of granularity ``G``, the score is
    ``cosine(probe, fingerprint_G[c])`` and ``IsTargetG = (probe true-G == c)``.

    * **KNOWN** scenario: keep only probes whose attack is KNOWN
      (``config.PROTOCOL.stopa_tee_attacks``); positives = target claims,
      negatives = different-known claims (row-major ``i``-then-``j`` order).
    * **UNKNOWN** scenario: positives = the true-target scores of KNOWN probes;
      negatives = *every* claim score of UNKNOWN-attack probes (the "unknown
      unknown" -- every unknown-attack probe must be rejected).

    EER via :func:`sourcetrace.metrics.protocol.stopa_eer` (symmetric ``(fpr+fnr)/2``
    -- deliberately NOT the asymmetric convention MLAAD uses; see
    :mod:`sourcetrace.metrics.protocol`).

    .. note::

       Scores are cast to ``float32`` before scoring. This is **load-bearing**:
       at ``float32`` precision many near-identical cosine scores collapse into
       exact ties, which changes which ROC vertex ``nanargmin`` selects and hence
       the reported EER. Do not widen the dtype.

    Args:
        trials_emb: ``(n_probe, emb_dim)`` probe embeddings.
        trials_attack: per-row probe attack label, row-aligned with ``trials_emb``.
        attacks_meta: ``{attack: {"am": ..., "vm": ...}}`` component metadata.
        fingerprints: output of :func:`build_fingerprints`.

    Returns:
        ``{G: {"known": eer_pct, "unknown": eer_pct}}`` for ``G`` in
        ``ATK``/``AM``/``VM``. An entry is ``nan`` when its scenario yields a
        single-class label vector.
    """
    scored = score_trials(trials_emb, fingerprints)
    known_set = set(KNOWN_ATTACKS)
    probe_is_known = np.array([a in known_set for a in trials_attack])

    true_G = {
        "ATK": np.array(trials_attack, dtype=object),
        "AM": np.array([attacks_meta[a]["am"] for a in trials_attack], dtype=object),
        "VM": np.array([attacks_meta[a]["vm"] for a in trials_attack], dtype=object),
    }

    results: dict[str, dict[str, float]] = {}
    for G in _GRANULARITIES:
        claims = scored[G]["claims"]
        S = np.asarray(scored[G]["scores"])
        tg = true_G[G]

        # is_tgt[i, j] -- does probe i's true G-label equal claim j?
        is_tgt = np.zeros_like(S, dtype=bool)
        for j, c in enumerate(claims):
            is_tgt[:, j] = (tg == c)

        # float32 is load-bearing for EER tie-breaking -- see the docstring note.
        S32 = S.astype(np.float32, copy=False)

        # KNOWN: all claim scores of known-attack probes (row-major i-then-j).
        k_scores = S32[probe_is_known].ravel()
        k_labels = is_tgt[probe_is_known].ravel().astype(np.int8)
        eer_known, _ = stopa_eer(k_labels, k_scores)

        # UNKNOWN: true-target scores of known probes (pos) + all claim scores of
        # unknown-attack probes (neg) -- every unknown-attack probe must be
        # rejected against every claim.
        pos_mask = probe_is_known[:, None] & is_tgt
        u_pos_scores = S32[pos_mask]
        u_neg_scores = S32[~probe_is_known].ravel()
        u_scores = np.concatenate([u_pos_scores, u_neg_scores])
        u_labels = np.concatenate([
            np.ones(u_pos_scores.shape[0], dtype=np.int8),
            np.zeros(u_neg_scores.shape[0], dtype=np.int8),
        ])
        eer_unknown, _ = stopa_eer(u_labels, u_scores)

        # stopa_eer already returns nan for degenerate inputs, and nan * 100 is
        # nan, so no explicit nan branch is needed here.
        results[G] = {
            "known": eer_known * 100.0,
            "unknown": eer_unknown * 100.0,
        }
    return results


def _stack_by_attack(table: Any, features: FeatureSource) -> tuple[np.ndarray, list[str]]:
    """Stack a STOPA table's features and its per-row attack labels, aligned.

    Rows are re-ordered by ``sorted(attack)``, stable within each attack (i.e.
    original table order is preserved inside a group). Features and labels are
    emitted in that same order so they stay row-aligned -- this matches the
    original worker's ``sorted(attack)`` cache load order
    (``stopa_verif.py`` L491-505).

    Args:
        table: a STOPA table exposing per-row ``attack`` and ``group`` lists.
        features: a :data:`FeatureSource` resolving each group's cached features.

    Returns:
        ``(features[n_rows, feat_dim], attack_labels)`` in sorted-attack order.
    """
    # Bucket row indices by attack in a single pass, then concatenate the buckets
    # in sorted-attack order. Equivalent to a stable sort on the attack key.
    buckets: dict[str, list[int]] = {}
    for i, atk in enumerate(table.attack):
        buckets.setdefault(atk, []).append(i)
    idx_by_attack = [i for atk in sorted(buckets) for i in buckets[atk]]

    ordered_groups = [table.group[i] for i in idx_by_attack]
    ordered_attack = [table.attack[i] for i in idx_by_attack]

    emb = stack_features(SimpleNamespace(group=ordered_groups), features)
    return emb, ordered_attack


def fit_stopa(
    method: Any,
    features: FeatureSource,
    tables: StopaTables | None = None,
    cond_filter: Any = None,
    seed: int | None = None,
) -> Any:
    """Fit a source-tracing method on the STOPA *EET* split (no eval).

    Train entry point for the split train/inference workflow. Fits ONLY on EET
    (leak-safe: TEE/Trials never enter ``fit``). Returns the fitted ``method`` so
    it can be checkpointed and later handed to :func:`eval_stopa`.

    Args:
        method: a method implementing ``fit(features, gen_labels, lang_labels, seed)``.
        features: a :data:`FeatureSource` resolving each table's cached features.
        tables: prebuilt :class:`StopaTables`; built when ``None``.
        cond_filter: passed to ``build_tables`` when ``tables`` is None (default
            ``None`` pools all conditions).
        seed: fit seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        The fitted ``method``.
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    tables = tables if tables is not None else build_tables(cond_filter=cond_filter)

    # -- EET fit labels: gen = attack id, lang = acoustic-model nuisance id ----- #
    eet_emb_src, eet_attack = _stack_by_attack(tables.eet, features)
    eet_gen = np.array([tables.atk2id[a] for a in eet_attack], dtype=np.int64)
    eet_lang = np.array([tables.am2id[tables.attacks_meta[a]["am"]] for a in eet_attack],
                        dtype=np.int64)
    method.fit(eet_emb_src, eet_gen, eet_lang, seed)
    return method


def eval_stopa(
    method: Any,
    features: FeatureSource,
    tables: StopaTables | None = None,
    cond_filter: Any = None,
    seed: int | None = None,
) -> StopaResult:
    """Evaluate a PRE-FIT source-tracing method on the STOPA verification protocol.

    Inference entry point: assumes ``method`` is already fitted (via
    :func:`fit_stopa` or :meth:`sourcetrace.method.Method.load`) and does NOT call
    ``fit``. Builds TEE fingerprints, scores Trials probes, and returns the 6 EERs.

    Args:
        method: a fitted method implementing ``embed(X) -> (N, emb_dim)``.
        features: a :data:`FeatureSource` resolving each table's cached features.
        tables: prebuilt :class:`StopaTables`; built when ``None``.
        cond_filter: passed to ``build_tables`` when ``tables`` is None.
        seed: unused here (fit already done); kept for signature parity.

    Returns:
        A :class:`StopaResult` with the headline unknown-attack ATK EER and all 6
        EERs (percentages).
    """
    tables = tables if tables is not None else build_tables(cond_filter=cond_filter)

    # -- embed TEE (enroll) + Trials (probe) in the aligned sorted-attack order - #
    tee_feat, tee_attack = _stack_by_attack(tables.tee, features)
    trials_feat, trials_attack = _stack_by_attack(tables.trials, features)
    tee_emb = np.asarray(method.embed(tee_feat))
    trials_emb = np.asarray(method.embed(trials_feat))

    fps = build_fingerprints(tee_emb, tee_attack, tables.attacks_meta)
    eers = compute_six_eers(trials_emb, trials_attack, tables.attacks_meta, fps)

    return StopaResult(
        unknown_atk_eer=eers["ATK"]["unknown"],
        six_eers=eers,
        n_tee=len(tee_emb),
        n_trials=len(trials_emb),
        n_fingerprints={G: len(fps[G]) for G in fps},
    )


def evaluate_stopa(
    method: Any,
    features: FeatureSource,
    tables: StopaTables | None = None,
    cond_filter: Any = None,
    seed: int | None = None,
) -> StopaResult:
    """Fit + evaluate a source-tracing method on the STOPA verification protocol.

    Back-compatible combined entry point: fits on EET then evaluates on TEE/Trials
    in one call. Equivalent to ``eval_stopa(fit_stopa(method, features, tables,
    cond_filter, seed), features, tables, cond_filter)``. The metric math is
    unchanged.

    Args:
        method: a method object implementing
            ``fit(features, gen_labels, lang_labels, seed)`` /
            ``embed(X) -> (N, emb_dim)``. STOPA scores by cosine-to-fingerprint on
            ``embed`` only; ``score_openset`` is not used here.
        features: a :data:`FeatureSource` resolving each table's per-group cached
            ``.npy`` features.
        tables: prebuilt :class:`StopaTables`; built via
            :func:`sourcetrace.datasets.stopa.build_tables` when ``None``.
        cond_filter: passed to ``build_tables`` when ``tables`` is None (default
            ``None`` pools all conditions).
        seed: fit seed; defaults to ``config.PROTOCOL.panda_seed`` (42).

    Returns:
        A :class:`StopaResult` with the headline unknown-attack ATK EER and all 6
        EERs (percentages).
    """
    seed = PROTOCOL.panda_seed if seed is None else int(seed)
    tables = tables if tables is not None else build_tables(cond_filter=cond_filter)
    fit_stopa(method, features, tables, cond_filter, seed)
    return eval_stopa(method, features, tables, cond_filter, seed)
