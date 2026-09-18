#!/usr/bin/env python3
"""Evaluate a trained :class:`Method` on the MLAAD v5 and/or STOPA protocols.

Loads a fitted :class:`sourcetrace.method.Method` from a ``.pt`` checkpoint (or
fits one on the fly when no checkpoint is given), runs the split-aware evaluation
entry points on the cached features, and prints a results table comparing the
headline numbers to the published reference figures:

* MLAAD v5 -> PANDA FPR@95 (vs 3.36, Neamtu et al., same v5 splits and FPR95
  definition -- a like-for-like comparison).
* STOPA    -> unknown-attack ATK EER (vs 16.43, Chhibber et al.), printed as **n/c**:
  16.43 is a *zero-shot* system, fitted on no STOPA split, while this head is fitted on
  STOPA's EET split. Same metric, different setting, so the table shows the reference for
  context and refuses to compute a margin. The like-for-like anchors are STOPA's own
  trained baselines; see Sec. 3.3 of the paper.

* ``--task mlaad_v5`` / ``--task stopa`` / ``--task both``.
* ``--checkpoint PATH`` loads a pre-fit method (:meth:`Method.load`); omit it to
  fit-then-eval in one shot (the combined ``evaluate_*`` path).
* ``--json OUT`` also writes the results as JSON.

The feature cache dir must hold the ``<group>.npy`` files for the task's splits
(extract them first). It defaults to ``<ST_FEATURE_CACHE>/<task>`` per task.

Examples
--------
    python -m sourcetrace.evaluate --task both --checkpoint-mlaad checkpoints/mlaad_v5.pt \
        --checkpoint-stopa checkpoints/stopa.pt
    python -m sourcetrace.evaluate --task stopa --checkpoint checkpoints/stopa.pt --json out.json
    python -m sourcetrace.evaluate --task mlaad_v5     # fit-then-eval (no checkpoint)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sourcetrace.config import (
    CONFIG_NAME,
    PROTOCOL,
    SOTA_MLAAD_V5_FPR95,
    SOTA_STOPA_UNKNOWN_EER,
)
from sourcetrace.runtime import require_feature_cache, require_torch


def _load_or_new(checkpoint: str | None):
    """Return (method, was_loaded). Fits later if not loaded."""
    from sourcetrace.method import Method

    if checkpoint:
        print(f"[eval] loading checkpoint {checkpoint}")
        try:
            return Method.load(checkpoint), True
        except (FileNotFoundError, IsADirectoryError) as exc:
            # Method.load already explains what is wrong and how to fix it; an entry
            # point should deliver that as a message, not a traceback. Same shape as the
            # missing-feature-cache diagnostic in runtime.require_feature_cache.
            raise SystemExit(
                f"{exc}\n\n"
                "--- sourcetrace: no checkpoint to evaluate -----------------------------\n"
                "Omit --checkpoint to fit and evaluate in one run instead.\n"
                "-----------------------------------------------------------------------"
            ) from None
    return Method(), False


def _run_mlaad(cache_dir: Path, checkpoint: str | None,
               split_seed: int, fit_seed: int) -> dict:
    from sourcetrace.datasets.mlaad import build_split
    from sourcetrace.tasks.mlaad_v5 import eval_mlaad_v5, evaluate_mlaad_v5

    # Protocol split uses split_seed (42, eval n=9,620); the on-the-fly fit uses
    # fit_seed (0). A loaded checkpoint only needs the split.
    split = build_split(seed=split_seed)
    method, loaded = _load_or_new(checkpoint)
    if loaded:
        res = eval_mlaad_v5(method, cache_dir, split=split, seed=split_seed)
    else:
        print("[eval] no checkpoint: fit-then-eval on MLAAD v5 ...", flush=True)
        res = evaluate_mlaad_v5(method, cache_dir, split=split, seed=fit_seed)
    return {
        "fpr95": res.fpr95, "ood_eer": res.ood_eer, "id_acc": res.id_acc,
        "n_known_classes": res.n_known_classes,
        "n_eval": res.n_eval, "n_eval_id": res.n_eval_id, "n_eval_ood": res.n_eval_ood,
    }


def _run_stopa(cache_dir: Path, checkpoint: str | None,
               split_seed: int, fit_seed: int) -> dict:
    from sourcetrace.datasets.stopa import build_tables
    from sourcetrace.tasks.stopa import eval_stopa, evaluate_stopa

    tables = build_tables()
    method, loaded = _load_or_new(checkpoint)
    if loaded:
        res = eval_stopa(method, cache_dir, tables=tables, seed=split_seed)
    else:
        print("[eval] no checkpoint: fit-then-eval on STOPA ...", flush=True)
        res = evaluate_stopa(method, cache_dir, tables=tables, seed=fit_seed)
    return {
        "unknown_atk_eer": res.unknown_atk_eer, "six_eers": res.six_eers,
        "n_tee": res.n_tee, "n_trials": res.n_trials,
    }


def _print_table(rows: list[tuple[str, str, float, float, bool]]) -> None:
    """rows: (task, metric, value, reference, comparable). Percentages.

    ``comparable`` is what stops this table lying. A WIN/DELTA verdict only means
    something when our number and the reference were produced under the same protocol:

    * **MLAAD v5** -- Neamtu et al.'s 3.36 is the same v5 splits and the same FPR95
      definition, so the delta is a margin and WIN is a verdict.
    * **STOPA** -- Chhibber et al.'s 16.43 is a *zero-shot* system, fitted on no STOPA
      split, while this head is fitted on EET. The two numbers share a metric and a
      benchmark but not a setting, so a difference confounds method with setting. Printing
      "WIN" there would hand a user a conclusion the paper itself declines to draw
      (Sec. 3.3), which a caveat in a docstring does not undo -- the terminal output is
      what people screenshot.

    Non-comparable rows therefore print the reference for context and ``n/c`` in place of
    a verdict, with the delta suppressed.
    """
    print()
    print(f"{'TASK':<10} {'METRIC':<22} {'OURS':>8} {'REF':>8} {'DELTA':>8}  RESULT")
    print("-" * 68)
    any_nc = False
    for task, metric, val, ref, comparable in rows:
        if comparable:
            verdict = "WIN " if val <= ref else "----"  # lower is better
            delta = f"{val - ref:>+8.2f}"
        else:
            any_nc, verdict, delta = True, "n/c ", f"{'--':>8}"
        print(f"{task:<10} {metric:<22} {val:>8.2f} {ref:>8.2f} {delta}  {verdict}")
    print("-" * 68)
    print("(lower is better; WIN means at or below the published reference)")
    if any_nc:
        print("(n/c = not comparable: same metric, different setting -- no margin claimed;")
        print(" see Sec. 3.3 of the paper)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sourcetrace.evaluate",
        description="Evaluate a trained Method on MLAAD v5 / STOPA vs SOTA.")
    # Read at import time by sourcetrace.config, not here -- see the note in that
    # module. Declared so it appears in --help and so a stray value is rejected.
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="hyperparameter overlay, e.g. configs/ablation_no_codec.yaml "
                             "(default: the frozen champion values, == configs/base.yaml)")
    parser.add_argument("--task", required=True, choices=["mlaad_v5", "stopa", "both"])
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint for the single-task modes (mlaad_v5|stopa)")
    parser.add_argument("--checkpoint-mlaad", default=None,
                        help="MLAAD v5 checkpoint (used in --task both)")
    parser.add_argument("--checkpoint-stopa", default=None,
                        help="STOPA checkpoint (used in --task both)")
    parser.add_argument("--cache-dir-mlaad", default=None,
                        help="MLAAD feature cache (default: <ST_FEATURE_CACHE>/mlaad_v5)")
    parser.add_argument("--cache-dir-stopa", default=None,
                        help="STOPA feature cache (default: <ST_FEATURE_CACHE>/stopa)")
    # Two separate seeds (see sourcetrace/train.py): the protocol split seed (42) and
    # the on-the-fly fit seed (0). With a loaded checkpoint only the split matters.
    parser.add_argument("--split-seed", type=int, default=None,
                        help="protocol split seed (default: PROTOCOL.panda_seed=42, "
                             "eval n=9,620)")
    parser.add_argument("--fit-seed", "--seed", dest="fit_seed", type=int, default=0,
                        help="fit seed for the no-checkpoint fit-then-eval path "
                             "(default: 0 -> the measured MLAAD v5 FPR95 of 0.50%%)")
    parser.add_argument("--json", default=None, help="write results JSON to this path")
    args = parser.parse_args(argv)

    # Diagnose the wrong interpreter before any download or dataset scan: a system
    # python with no torch (or a stub one) otherwise fails much later, from inside a
    # feature extractor, with an error that never mentions the interpreter.
    require_torch()

    split_seed = PROTOCOL.panda_seed if args.split_seed is None else args.split_seed
    fit_seed = args.fit_seed
    results: dict = {}
    table: list[tuple[str, str, float, float]] = []

    if args.task in ("mlaad_v5", "both"):
        ckpt = args.checkpoint_mlaad or (args.checkpoint if args.task == "mlaad_v5" else None)
        cdir = require_feature_cache("mlaad_v5", args.cache_dir_mlaad)
        print(f"[eval] MLAAD v5 features: {cdir}")
        r = _run_mlaad(cdir, ckpt, split_seed, fit_seed)
        results["mlaad_v5"] = r
        table.append(("mlaad_v5", "FPR@95", r["fpr95"], SOTA_MLAAD_V5_FPR95, True))

    if args.task in ("stopa", "both"):
        ckpt = args.checkpoint_stopa or (args.checkpoint if args.task == "stopa" else None)
        cdir = require_feature_cache("stopa", args.cache_dir_stopa)
        print(f"[eval] STOPA features: {cdir}")
        r = _run_stopa(cdir, ckpt, split_seed, fit_seed)
        results["stopa"] = r
        # not comparable: 16.43 is zero-shot, this head is fitted on EET (see _print_table)
        table.append(("stopa", "unknown-ATK EER", r["unknown_atk_eer"],
                      SOTA_STOPA_UNKNOWN_EER, False))

    _print_table(table)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            # `config` records which configs/*.yaml produced these numbers, or
            # null for the frozen defaults. Before configs/ existed, an ablation
            # arm was an ST_ABL_* string in a shell array and the result file said
            # nothing about what it had been fit under; the files committed under
            # results/ predate this key and do not carry it.
            json.dump({"split_seed": split_seed, "fit_seed": fit_seed,
                       "config": CONFIG_NAME, "results": results,
                       "sota": {"mlaad_v5_fpr95": SOTA_MLAAD_V5_FPR95,
                                "stopa_unknown_atk_eer": SOTA_STOPA_UNKNOWN_EER}},
                      fh, indent=2)
        print(f"[eval] wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
