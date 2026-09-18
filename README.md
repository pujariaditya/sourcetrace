<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/hero-dark.png">
    <img alt="sourcetrace: name the generator behind a clip, or report it unknown." src="assets/readme/hero-light.png" width="840">
  </picture>
</p>

# sourcetrace

*CORE: Enhancing Open-Set Audio Deepfake Attribution with Neural Codec Residuals*

![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)
[![Checkpoints](https://img.shields.io/badge/%F0%9F%A4%97-Checkpoints-ffcc4d)](https://huggingface.co/RootAccess4Life/ood-source-tracing)

> 📦 Checkpoints: **[RootAccess4Life/ood-source-tracing](https://huggingface.co/RootAccess4Life/ood-source-tracing)**
> — [`mlaad_v5.pt`](https://huggingface.co/RootAccess4Life/ood-source-tracing/blob/main/mlaad_v5.pt)
> is the artifact of record for every number below &nbsp;·&nbsp;
> [`stopa.pt`](https://huggingface.co/RootAccess4Life/ood-source-tracing/blob/main/stopa.pt)
> is the second corpus.
> Both are sha256-pinned in `scripts/download_weights.py`; fetch with
> `python scripts/download_weights.py`.

**Given a synthetic speech clip, name which generator produced it — or report that
the generator is not one it has seen.** That second answer is the point: new systems
appear constantly, and an attribution nobody can trust is worse than none.

A **frozen** WavLM-Large front-end (layers 1–4, mean|std pooled), no fine-tuning, a
52-d spectral phase/modulation signature, and a small trained head. The distinguishing
ingredient is a **codec reconstruction residual**: squeeze a clip through EnCodec-24 kHz,
pull it back, and keep what it got *wrong*. Those 33 numbers come off the waveform the
speech model never sees; a small trained projection folds them into the embedding.

**FPR95 0.50 % against a published 3.36 % · OOD-EER 2.45 % · closed-set accuracy
99.38 % · STOPA unknown-attack EER 9.27 %.** The residual channel is worth 0.78 FPR95
points and 1.0 OOD-EER points on MLAAD v5; on STOPA it is neutral (9.31 % without it),
which the paper says rather than hides ([table](#results)).

```bash
git clone https://github.com/pujariaditya/sourcetrace && cd sourcetrace
conda create -n sourcetrace python=3.10 && conda activate sourcetrace
pip install -e . && ./setup.sh
python scripts/smoke.py              # 20 s, no download: check the install
```

Export `PYTHONNOUSERSITE=1` in that environment — a newer `huggingface_hub` in
`~/.local` shadows the pinned one and breaks the pinned `transformers`. Run from the
repository root; every entry point checks the interpreter first and says so when it is
wrong.

`smoke.py` exercises fit → embed → open-set score → save → reload on a tiny synthetic
matrix, so a broken install fails in seconds instead of after a 242 GB download. It
says nothing about accuracy; `./scripts/verify_results.sh` is what does.

## Results

On **MLAAD v5**, family-level open set: 65 known families, 9,620 evaluation trials,
split seed 42 / fit seed 0.

| metric | this repo | published |
|---|---|---|
| FPR95 ↓ | **0.50 %** | 3.36 % |
| OOD-EER ↓ | 2.45 % | — |
| closed-set accuracy ↑ | 99.38 % | — |

The published reference is PANDA / Neamtu et al.
([arXiv:2606.10758](https://arxiv.org/abs/2606.10758)) on the official v5 splits under
the same FPR95 definition. They report no OOD-EER or closed-set accuracy, which is why
those rows have no reference rather than a favourable one.

**The codec channel is worth 0.78 FPR95 points and 1.0 OOD-EER points.** Deleting it
moves FPR95 from 0.50 to 1.28 and OOD-EER from 2.45 to 3.45, at unchanged closed-set
accuracy (`results/ablation/no_codec.json`). Replacing the residual by noise or by
shuffled residuals gives 1.33 and 1.76, so the gain comes from the residual content.
The projection that folds the 33 residual features into the embedding is trained with
the head; leaving it at its random initialisation, the earlier design, costs 0.28 FPR95
points and 0.9 OOD-EER points (`results/ablation/fixed_proj.json`).

**The SSL layer band matters and is chosen on development data.** Layers 1–4 score 0.50
FPR95 against 0.91 for 4–7 and 1.58 for 8–11, and the development split orders the bands
the same way (OOD-EER 1.95 / 2.11 / 3.16), so the choice does not read the evaluation
split (`results/layer_band_summary.json`).

The conformal abstention rule reports its measured coverage against the 95 % nominal
level in `results/reliability.json`.

On **STOPA** — a second corpus, 629,800 probe utterances, five unknown attacks — the
same front end and hyperparameters, with the head refitted on STOPA's EET split, score
**9.27 %** unknown-attack EER against mid-30s to near 50 % for STOPA's own trained
baselines. **No margin is claimed over the lower published 16.43 %**: that is a
*zero-shot* system, a different setting, and `evaluate` prints `n/c` rather than a win
(`results/stopa_measured.json`).

Every number is single-seed. None of these margins is a measured effect size.

## Reproduce the table

```bash
python scripts/download_weights.py                        # both heads, sha256-pinned
python scripts/fetch_mlaad.py                             # ~242 GB of audio
python -m sourcetrace.extract --dataset mlaad_v5          # audio -> 2133-d features
python -m sourcetrace.evaluate --task mlaad_v5 \
    --checkpoint checkpoints/mlaad_v5.pt --json runs/my_run.json
```

Write your own runs to `runs/`, not `results/` — that directory is the committed
record the paper cites, and a re-run that overwrote it in place would leave no way to
tell a reproduction from the original.

Downloading a checkpoint saves the ~35–40 minute fit and nothing else; evaluation
still reads the feature cache built above. To refit instead, drop `--checkpoint`, or
use `python -m sourcetrace.train`. `./scripts/verify_results.sh` runs the whole chain
end to end, and `./scripts/run_ablations.sh` reproduces the ablation table
(one ~35–40 min fit per arm).

Paths are environment-driven, never hardcoded: `ST_DATA_ROOT`, `ST_MLAAD_ROOT`,
`ST_STOPA_ROOT`, `ST_FEATURE_CACHE`, `ST_MODEL_CACHE`, and `ST_CPU=1` to force CPU.
Hyperparameters come from `configs/` — `configs/base.yaml` is the frozen configuration
and is checked to be identical to the code's own defaults; `train_codec_path: false`
there reproduces the earlier fixed-projection system.

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the benchmarks — MLAAD
([mueller91/MLAAD](https://huggingface.co/datasets/mueller91/MLAAD)) and STOPA
([10.5281/zenodo.15606628](https://doi.org/10.5281/zenodo.15606628)) — and the
baselines compared against: Neamtu et al.
([arXiv:2606.10758](https://arxiv.org/abs/2606.10758)) and Chhibber et al.
([arXiv:2509.24674](https://arxiv.org/abs/2509.24674)).

## Licence

MIT (`LICENSE`). MLAAD and STOPA carry their own terms; this repository redistributes
no audio and no weights.
