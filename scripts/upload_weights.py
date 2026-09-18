#!/usr/bin/env python3
"""Publish the two fitted heads to Hugging Face, and refresh the model card.

    HF_TOKEN=... python scripts/upload_weights.py --mlaad_v5 checkpoints/mlaad_v5.pt \
                                                  --stopa    checkpoints/stopa.pt
    HF_TOKEN=... python scripts/upload_weights.py --card-only

Maintainer tool; end users want ``download_weights.py``.

THE TOKEN IS READ FROM THE ENVIRONMENT ONLY. It is never written to a config, a
committed file, or the upload itself. ``huggingface-cli login`` would persist it
to ``~/.cache/huggingface/token``, which is world-readable by default -- pass it
per-invocation instead.

Every file is hashed before upload and the digest printed. Those are the values
that belong in ``download_weights.py``; the point of pinning them there is that a
reader verifies against a constant in the source, never against a digest served
from the same place as the file.

The card lives here, as a constant, rather than being edited on the Hub. It was
edited on the Hub for a while and drifted: it went on instructing a download
command and an evaluation command that the repository had renamed, and it was the
last place still printing two retired figures. A card with no source in the
repository cannot be reviewed in a diff.
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sourcetrace.runtime import require_packages

REPO = os.environ.get("ST_HF_REPO", "RootAccess4Life/ood-source-tracing")

CARD = """---
license: mit
library_name: pytorch
pipeline_tag: audio-classification
tags:
  - audio
  - audio-classification
  - deepfake-detection
  - source-tracing
  - open-set-recognition
  - wavlm
  - encodec
datasets:
  - mueller91/MLAAD
metrics:
  - accuracy
---

# sourcetrace — fitted heads for open-set audio deepfake source tracing

Fitted heads for [`sourcetrace`](https://github.com/pujariaditya/sourcetrace), which
names **which generator produced a synthetic speech clip** — or reports that the
generator is not one it has seen.

**These are not standalone models.** Each file is a `torch.save` dict with
`format: "sourcetrace-method-checkpoint"`, loaded by `sourcetrace.method.Method.load`.
They hold the small trained head plus the fitted scoring stack (class anchors,
relative-Mahalanobis density, z-norm constants, conformal calibration, per-block
whitening). The front-ends — `microsoft/wavlm-large` and `facebook/encodec_24khz` — are
frozen, are **not** included here, and are fetched separately by `./setup.sh`.

## Model

| | |
|---|---|
| Input | 2133-d feature vector, **not** audio |
| SSL front end | `microsoft/wavlm-large`, **frozen**, layers 1–4, mean\\|std pooled → 2048-d |
| Signature channel | 24 group-delay bands + 16 modulation bins + 12 codec-grid dims → 52-d |
| Codec channel | `facebook/encodec_24khz` reconstruction residual at 1.5 / 6 / 12 kbps → 33-d |
| Head | factorized gated, AM 192 + vocoder 64 + codec 32 = 288-d, tanh FiLM gates |
| Embedding | 800-d = 288 score branch ‖ 512 whitened transfer branch |
| Trainable parameters | 1,089,888 (548,576 head + 541,312 transfer) |
| Training | Adam, lr 3e-4, no weight decay, 300 epochs, batch 512 |

The file is far larger than the parameter count because most of it is the fitted
scoring stack — 65 per-class Mahalanobis precisions at 288×288 float64, plus the
whitening matrices — not weights.

The codec channel is the contribution: re-encode each clip through EnCodec-24 kHz
and keep what it got *wrong*. Those 33 numbers come off the waveform the speech
model never sees. Deleting the channel moves FPR95 from 0.50 % to 1.28 % and OOD-EER from 2.45 % to 3.45 % while
closed-set accuracy barely changes — it buys *rejection*, not attribution.

The head in the public code is a reconstruction from the published method
description, not a recovered original; its construction order is load-bearing for
RNG reproducibility.

## Input

Not audio. A **2133-d** feature vector per clip, laid out as
`[ SSL 0:2048 | signature 2048:2100 | codec residual 2100:2133 ]`, produced by
`python -m sourcetrace.extract`. There is no way to run these weights without the
repository and an extracted feature cache.

## Files

| file | size | what it is |
|---|---|---|
| `mlaad_v5.pt` | 79 MB | MLAAD v5, 65 known generator families |
| `stopa.pt` | 18 MB | STOPA, 13 attack classes |

Both are sha256-verified on download against digests compiled into
`scripts/download_weights.py`, not served from here. A digest served from the same
place as the file it describes attests to nothing.

## Use

```bash
git clone https://github.com/pujariaditya/sourcetrace && cd sourcetrace
pip install -e . && ./setup.sh
python scripts/download_weights.py      # both heads, sha256-pinned
python -m sourcetrace.evaluate --task mlaad_v5 --checkpoint checkpoints/mlaad_v5.pt
```

Downloading saves the ~35–40 min fit and nothing else: evaluation still reads the
feature cache, so MLAAD v5 must be downloaded and extracted first. See the
repository README for that step.

## `mlaad_v5.pt`

MLAAD v5, family-level open-set protocol: 65 known generator families, 9,620 evaluation
trials, split seed 42, fit seed 0.

| Metric | Value | Published SOTA |
|---|---|---|
| FPR95 ↓ | **0.50 %** | 3.36 % |
| OOD-EER ↓ | 2.45 % | — |
| Closed-set accuracy ↑ | 99.38 % | — |

Conformal abstention: measured coverage 96.11 % against a 95 % nominal level.

**Reproducible, not just reported.** Refitting from the public code at these seeds
reproduces `results/ablation/full.json` bit-for-bit — FPR95 0.5028571428571428, OOD-EER
2.4457142857142857, closed-set 99.38285714285713, under exact equality rather than a
tolerance. This file is that fit.

**Single-seed.** One split seed, one fit seed. These are point estimates with no
variance attached; do not read the margin over 3.36 % as a measured effect size.

## `stopa.pt`

STOPA cross-corpus open-set protocol: 3 EET attacks fitted, 5 known enrolled, 5 unknown
held out; 33,200 enrolment and 629,800 probe utterances, conditions pooled, split seed
42 / fit seed 0.

| Metric | Value | Published |
|---|---|---|
| Unknown-attack EER ↓ | **9.27 %** | 16.43 % |
| Known-attack EER ↓ | 9.65 % | — |

Measured by the released code on 2026-09-18; raw numbers in
`results/stopa_measured.json`.

9.27 % sits below every baseline STOPA itself reports — mid-30s for AASIST-family
countermeasures, near 50 % for ResNet-34 — and those are the like-for-like anchors:
trained systems, scored the same way, over the same partition. **No margin is claimed
over the lower published 16.43 %.** That is a *zero-shot* system, fitted on no STOPA
split, where this head is fitted on EET; the two share a metric but not a setting.

**Single-seed**, like the MLAAD numbers above.

## Limitations

- Trained on MLAAD v5 only. Attribution across other corpora, languages, codecs or
  recording conditions is untested.
- Both numbers are single-seed point estimates.
- For research on open-set attribution. **Not validated for forensic, legal or
  moderation use**, and the abstention rule is calibrated on this protocol — its
  coverage guarantee does not transfer off it.
- The harm from an attribution model is a confident wrong name, not a refusal. On an
  unseen generator the calibrated answer is *unknown*, and that answer is the point
  of the system; do not deploy it anywhere the abstention is discarded, and do not
  present an attribution as evidence about a person.

## Licence

MIT, matching the code repository. MLAAD and STOPA carry their own terms; no audio is
redistributed here.

## Citation

See [`CITATION.cff`](https://github.com/pujariaditya/sourcetrace/blob/main/CITATION.cff)
in the code repository, which is the single source for how to cite this.

Please also cite the benchmarks (MLAAD, STOPA) and the baselines this is compared
against (Neamtu et al., EUSIPCO 2026; Chhibber et al., Odyssey 2026).
"""


def sha256_of(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(prog="python scripts/upload_weights.py",
                                 description=__doc__.split("\n")[0])
    # All optional: re-uploading one head should not mean re-hashing and
    # re-pushing the other.
    ap.add_argument("--mlaad_v5", default=None, help="path to the fitted MLAAD v5 head")
    ap.add_argument("--stopa", default=None, help="path to the fitted STOPA head")
    ap.add_argument("--card-only", action="store_true",
                    help="refresh the model card and upload no checkpoints")
    ap.add_argument("--private", action="store_true", help="create the repo private")
    ap.add_argument("--dry-run", action="store_true", help="hash and report, upload nothing")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit(
            "set HF_TOKEN in the environment (do not use `huggingface-cli login`, "
            "which writes a world-readable token file)"
        )

    targets = [(os.path.basename(p), p)
               for p in (args.mlaad_v5, args.stopa) if p]
    if not targets and not args.card_only:
        raise SystemExit("nothing to upload: pass --mlaad_v5 / --stopa, or --card-only")
    if targets and args.card_only:
        raise SystemExit("--card-only uploads no checkpoints; drop the path arguments "
                         "or drop --card-only")

    plan = []
    for remote, local in targets:
        if not os.path.isfile(local):
            raise SystemExit(f"not a file: {local}")
        digest = sha256_of(local)
        print(f"| {remote}  {os.path.getsize(local) / 1e6:.1f} MB  sha256 {digest}")
        plan.append((local, remote))

    if plan:
        print("|")
        print("| paste these into CHECKPOINTS in scripts/download_weights.py")

    if args.dry_run:
        if args.card_only:
            print(CARD)
        print("| dry run: nothing uploaded")
        return 0

    require_packages("huggingface_hub")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(REPO, repo_type="model", private=args.private, exist_ok=True)

    for local, remote in plan:
        print(f"| uploading {remote}")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=REPO, repo_type="model")

    # The card describes every artifact, so refresh it whenever anything is
    # published -- a partial upload still changes what the repo contains.
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="model")
    print(f"| done: https://huggingface.co/{REPO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
