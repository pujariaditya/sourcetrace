#!/usr/bin/env python3
"""Fetch the released checkpoints and verify them by sha256.

    python scripts/download_weights.py                 # both heads
    python scripts/download_weights.py --task mlaad_v5 # just the headline one

Two heads are published, one per protocol. Neither is a standalone model: each is
a ``torch.save`` dict holding the small trained head plus the fitted scoring stack
(class anchors, relative-Mahalanobis density, z-norm constants, conformal
calibration, per-block whitening). The frozen front-ends are not here -- they are
fetched by ``setup.sh`` -- and evaluation still reads the extracted feature cache,
so downloading saves the fit and nothing else.

The hashes are compiled in rather than fetched alongside the files. A digest
served from the same place as the file it describes attests to nothing.

That is not a theoretical concern here. Until 2026-08-19 the Hub carried an
``mlaad_v5.pt`` that predated a refactor of the head, with nested proj/gate
submodules and a transposed whitening matrix. It did not fail loudly: a careful
hand-conversion of it scored FPR95 4.89 where this checkout scores 0.50. A wrong
checkpoint here produces a plausible number that is wrong by a factor of four,
not an error anyone would notice.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

# Importable without an install, so a fresh clone can fetch weights before
# `pip install -e .` has run.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_packages

REPO = os.environ.get("ST_HF_REPO", "RootAccess4Life/ood-source-tracing")

#: task -> (filename, sha256, what it is). Verified 2026-09-18 by downloading each
#: file back off the Hub, hashing it, loading it with Method.load and re-running
#: the evaluation.
CHECKPOINTS = {
    "mlaad_v5": (
        "mlaad_v5.pt",
        "96c6472d8801df8e60a3d5c49f14818ef293e3f0ecb38131d3749384a71d36a6",
        "MLAAD v5, 65 known families, WavLM layers 1-4, trained residual projection. Reproduces "
        "results/ablation/full.json bit-for-bit: FPR95 0.5028571428571428, OOD-EER 2.4457142857142857, "
        "closed-set 99.38285714285713, under exact equality rather than a tolerance.",
    ),
    "stopa": (
        "stopa.pt",
        "9229e0f3203579a8396c4b5afe3d2ecb4012bd6a5bd55ec43fbb2285c28284fa",
        "STOPA, 13 attack classes, same front end. Unknown-attack EER 9.2728%; raw numbers in "
        "results/stopa_measured.json.",
    ),
}

#: The broken upload, named so a stale copy is recognised rather than puzzled over.
_KNOWN_BAD = {
    "1bcddc3a": "the pre-refactor mlaad_v5.pt withdrawn on 2026-08-19; it loads "
                "and scores FPR95 4.89 instead of 0.50",
}


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(task, dest_root):
    """Download one checkpoint and refuse it unless the digest matches."""
    from huggingface_hub import hf_hub_download

    name, expected, note = CHECKPOINTS[task]
    print(f"==> {name}")
    print(f"|   {note}")

    cached = hf_hub_download(REPO, name)
    got = sha256_of(cached)
    if got != expected:
        hint = _KNOWN_BAD.get(got[:8])
        raise SystemExit(
            f"sha256 mismatch for {name}\n"
            f"  expected {expected}\n"
            f"  got      {got}\n"
            + (f"  that digest is {hint}.\n" if hint else "")
            + "Refusing to use a checkpoint that is not the released one."
        )
    print(f"|   sha256 ok: {got}")

    os.makedirs(dest_root, exist_ok=True)
    dest = os.path.join(dest_root, name)
    # Copy rather than symlink: checkpoints/ is what --checkpoint points at, and a
    # dangling link into a pruned HF cache is a worse failure than a duplicated file.
    if os.path.realpath(cached) != os.path.realpath(dest):
        import shutil
        shutil.copy(cached, dest)
    print(f"|   {dest}")
    return dest


def main():
    ap = argparse.ArgumentParser(prog="python scripts/download_weights.py",
                                 description=__doc__.split("\n")[0])
    ap.add_argument("--task", choices=sorted(CHECKPOINTS), default=None,
                    help="fetch only this head (default: both)")
    ap.add_argument("--dest", default="checkpoints",
                    help="destination directory (default: checkpoints)")
    ap.add_argument("--verify-only", action="store_true",
                    help="hash the local files against the compiled-in digests and "
                         "download nothing")
    args = ap.parse_args()

    # Before any network call, and after --help, so a broken interpreter gets one
    # readable line instead of a traceback from inside the hub client.
    if not args.verify_only:
        require_packages("huggingface_hub")

    tasks = [args.task] if args.task else sorted(CHECKPOINTS)

    if args.verify_only:
        bad = 0
        for task in tasks:
            name, expected, _ = CHECKPOINTS[task]
            path = os.path.join(args.dest, name)
            if not os.path.isfile(path):
                print(f"| {name}: absent")
                bad += 1
                continue
            got = sha256_of(path)
            ok = got == expected
            bad += not ok
            print(f"| {name}: {'ok' if ok else 'MISMATCH'}  {got}")
            if not ok and got[:8] in _KNOWN_BAD:
                print(f"|   that digest is {_KNOWN_BAD[got[:8]]}")
        return 1 if bad else 0

    for task in tasks:
        fetch(task, args.dest)

    print()
    print("Next:")
    print("  python -m sourcetrace.evaluate --task mlaad_v5 \\")
    print("      --checkpoint checkpoints/mlaad_v5.pt --json runs/my_run.json")
    print("  Expect FPR95 0.50. Evaluation reads the feature cache, which the")
    print("  download does not provide -- see the README for building it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
