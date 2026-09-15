"""Factorized gated embedding head + block-whitened PCA STOPA transfer branch.

RECONSTRUCTION NOTICE
---------------------
This module was reconstructed from the published method description and the interface
contract asserted throughout ``sourcetrace/method.py``, because the original
``sourcetrace/models/`` (commit ``fa876f7d``) was not included in the public checkout.
The architecture, dimensions, gates, subspace weights and forward math follow METHOD.md;
every design choice not pinned by it is marked ``# ASSUMED``.

RNG / INITIALIZATION ORDER IS LOAD-BEARING
------------------------------------------
Given a fixed seed, every ``nn.Module`` constructed here draws from the global torch RNG
in source order. Reordering, adding, or removing any submodule construction, parameter,
or ``_init_gate`` call shifts every subsequent draw and silently changes all downstream
results. Treat the construction sequence in :class:`FactorizedGatedHead.__init__` and in
:func:`build_head_pair` as frozen; edit the docs here, never the order.

FIXED RANDOM CODEC SUBSPACE
---------------------------
``proj_codec`` and ``gate_codec`` (score branch only) are **deliberately never added to
the optimizer** by ``Method.fit`` (see the EXCLUSIONS note in ``method.py``). They are
ordinary ``nn.Linear`` modules that receive gradients but are never stepped, so the codec
subspace is a **fixed random projection of the train-normalized codec residual for the
entire run** — its values are exactly whatever the seeded init produced. This is a
property of the method, not an oversight; do not "fix" it by adding them to ``params``.

Interface required by ``method.py``:
    build_head_pair(device) -> (score_head, transfer_head)
    FactorizedGatedHead(nn.Module):
        .proj_am / .proj_voc / .gate_voc / .gate_tmp   (proj_voc/gates may be None)
        .proj_codec / .gate_codec                      (score branch only; else None)
        .codec_mu / .codec_sd                          (non-persistent buffers)
        .set_codec_norm(mu, sd); forward(x[N, 2133]) -> [N, 288] (score) / [N, 256] (xfer)
    RawWhitenTransfer:
        .fit(x_all[N, 2133]); .apply(x_mean_half[N, 1024]) -> [N, 512]; .is_fit
        ._raw_mu / ._raw_Wz / ._perlayer   (checkpointed by method.py by these names)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FEATURES, MODEL

# feature-layout offsets (METHOD.md §1): [ SSL 0:2048 | sig 2048:2100 | codec 2100:2133 ]
_SSL = FEATURES.ssl_dim                       # 2048
_SIG = FEATURES.sig_dim                       # 52
_CODEC = FEATURES.codec_dim                   # 33
_SIG_OFF = _SSL                               # 2048
_CODEC_OFF = _SSL + _SIG                      # 2100
_HALF = _SSL // 2                             # 1024 (mean half | std half)


def _init_gate(lin: nn.Linear) -> None:
    """Near-identity gate init: ``N(0, MODEL.gate_init_std**2)`` weight, zero bias.

    Small-random rather than exactly zero, so the gate units are not symmetric: an
    all-zero weight makes every unit of ``lin`` see the same gradient and stay tied to
    its neighbours. The *near-identity* property comes from ``tanh(0) = 0``, which
    leaves ``z * (1 + beta * tanh(g)) == z`` at initialisation, so training begins from
    a clean concatenation and the modulation is learned on top of it.

    Note the reason to avoid an exact zero is symmetry, not vanishing gradient:
    ``tanh'(0) = 1`` is the *maximal* slope, so gradient flows fine either way.
    See METHOD.md §2.
    """
    nn.init.normal_(lin.weight, mean=0.0, std=MODEL.gate_init_std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


class FactorizedGatedHead(nn.Module):
    """Project the three frozen channels into gated subspaces and concatenate.

    Score branch (``codec=True``)  -> ``L2([z_am(192) | 0.1 z_voc(64) | 0.5 z_codec(32)])`` = 288-d.
    Transfer branch (``codec=False``) -> ``L2([z_am(192) | 0.1 z_voc(64)])`` = 256-d.

    The submodules below are constructed in a fixed order that determines the seeded
    RNG draw sequence; see the module-level "RNG / INITIALIZATION ORDER IS LOAD-BEARING"
    notice. Do not reorder them.
    """

    def __init__(self, codec: bool) -> None:
        super().__init__()
        self.codec = codec

        # AM subspace from the (tmp-gated) SSL vector.
        self.proj_am = nn.Linear(_SSL, MODEL.fact_am_dim)                 # 2048 -> 192
        # temporal-std FiLM: mean-half modulates std-half, bottleneck 1024->64->1024.
        self.gate_tmp = nn.Sequential(                                    # ASSUMED: ReLU inner
            nn.Linear(_HALF, MODEL.tmp_gate_hidden),
            nn.ReLU(),
            nn.Linear(MODEL.tmp_gate_hidden, _HALF),
        )
        _init_gate(self.gate_tmp[0])
        _init_gate(self.gate_tmp[2])

        # vocoder subspace from the 52-d signature, AM->voc FiLM gate.
        self.proj_voc = nn.Linear(_SIG, MODEL.fact_voc_dim)              # 52 -> 64
        self.gate_voc = nn.Linear(MODEL.fact_am_dim, MODEL.fact_voc_dim)  # 192 -> 64
        _init_gate(self.gate_voc)

        # Codec subspace (score branch only).  NEVER added to the optimizer by
        # Method.fit, so both of these Linears keep their seeded initialization for the
        # whole run: the codec subspace is a FIXED RANDOM PROJECTION of the
        # train-normalized codec residual.  Intentional — see the module docstring.
        if codec:
            self.proj_codec = nn.Linear(_CODEC, MODEL.fact_codec_dim)    # 33 -> 32
            self.gate_codec = nn.Linear(MODEL.fact_am_dim, MODEL.fact_codec_dim)  # 192 -> 32
            _init_gate(self.gate_codec)
        else:
            self.proj_codec = None
            self.gate_codec = None

        # codec-residual TRAIN normalization buffers (non-persistent; set in fit()).
        self.register_buffer("codec_mu", None, persistent=False)
        self.register_buffer("codec_sd", None, persistent=False)

    # -- codec-residual train-normalization ---------------------------------- #
    def set_codec_norm(self, mu: torch.Tensor | None, sd: torch.Tensor | None) -> None:
        self.codec_mu = mu
        self.codec_sd = sd

    # -- forward -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ssl = x[:, :_SSL]
        mean, std = ssl[:, :_HALF], ssl[:, _HALF:]
        # temporal-std FiLM (beta = tmp_gate_beta), bounded by tanh (METHOD.md §2).
        std = std * (1.0 + MODEL.tmp_gate_beta * torch.tanh(self.gate_tmp(mean)))
        ssl = torch.cat([mean, std], dim=1)

        z_am = F.normalize(self.proj_am(ssl), p=2, dim=1)                 # [N, 192]
        parts = [z_am]

        # vocoder subspace, AM->voc FiLM (beta = voc_gate_beta).
        sig = x[:, _SIG_OFF:_SIG_OFF + _SIG]
        z_voc = F.normalize(self.proj_voc(sig), p=2, dim=1)              # [N, 64]
        z_voc = z_voc * (1.0 + MODEL.voc_gate_beta * torch.tanh(self.gate_voc(z_am)))
        parts.append(MODEL.voc_weight * z_voc)

        # codec subspace (score branch), AM->codec FiLM (beta = codec_gate_beta).
        if self.codec and self.proj_codec is not None:
            cr = x[:, _CODEC_OFF:_CODEC_OFF + _CODEC]
            if self.codec_mu is not None and self.codec_sd is not None:
                cr = (cr - self.codec_mu) / self.codec_sd                 # TRAIN-normalized
            z_codec = F.normalize(self.proj_codec(cr), p=2, dim=1)        # [N, 32]
            z_codec = z_codec * (1.0 + MODEL.codec_gate_beta * torch.tanh(self.gate_codec(z_am)))
            parts.append(MODEL.codec_weight * z_codec)

        return F.normalize(torch.cat(parts, dim=1), p=2, dim=1)


def build_head_pair(device: torch.device | str):
    """Build the score-branch (codec=True) and transfer-branch (codec=False) heads.

    The score head is constructed first, then the transfer head. This order is ASSUMED
    (the champion's exact submodule/RNG order is not recoverable, so bit-identical
    seeded init is not guaranteed — see the module RECONSTRUCTION NOTICE), but it is
    now fixed: both heads draw from the same global torch RNG, so swapping the two
    lines below re-rolls every weight in both heads and changes all results. Do not
    reorder them.

    The two heads are **independent objects sharing no parameters**. The transfer head
    is trained by the auxiliary loss and then never read at inference -- ``Method.embed``
    builds the transfer half from :class:`RawWhitenTransfer` instead -- so no gradient
    from the transfer loss reaches the score branch and its trained weights influence no
    reported number. See ``Method.embed``.

    IT IS STILL CONSTRUCTED, AND DELETING IT CHANGES THE RESULTS. Not the score head,
    which is built on the line above with its initial weights already drawn, and not the
    per-class anchors, which ``Method.fit`` draws from dedicated CPU generators seeded 0
    and 1, independent of the global stream. What it moves is everything the global RNG
    serves *after* this point --
    in particular the HamOS virtual-outlier draw
    (``losses/objectives.py``, which documents its dependence on the global stream), once
    per batch per epoch. Different outliers, different fit, different reported numbers.
    Since ``results/`` was measured with this call present, removing it is not a
    simplification; it is a silent invalidation of every number the paper reports.
    """
    score = FactorizedGatedHead(codec=True).to(device)
    # DEAD AT INFERENCE, load-bearing for the RNG draw: see the note above.
    xfer = FactorizedGatedHead(codec=False).to(device)
    return score, xfer


# --------------------------------------------------------------------------- #
# STOPA transfer channel: blockwise-whitened + PCA-compressed WavLM mean-half -> 512-d
# --------------------------------------------------------------------------- #


class RawWhitenTransfer:
    """Blockwise full-rank whitening of the 1024-d mean-half, then PCA-compress to 512-d.

    Procedure (METHOD.md §3): split the 1024-d mean-half into 4 contiguous 256-d blocks,
    full-rank whiten each block via economy SVD of the centred block, concatenate, then
    PCA-compress to 512-d (no second whitening). All statistics are fit on train and
    applied thereafter as constants. Only affects STOPA / the 800-d embed; MLAAD scoring
    slices ``[:288]`` and never sees this channel.

    MINIMUM FIT SIZE. Both SVDs are economy-sized, so ``fit`` needs at least
    ``max(block_dim, raw_whiten_compress)`` = 512 rows to reach the configured widths;
    below that the bases come out narrower and ``apply`` silently returns fewer than
    512 columns. ``fit`` raises :class:`ValueError` rather than allow that, because the
    consequence otherwise appears far downstream as an opaque width rejection in
    ``Method.score_openset``. The real fit sets are far above this bound (MLAAD v5 train
    is 20,401 rows, STOPA EET 48,024), so the guard only ever fires on toy data.

    THE FOUR BLOCKS ARE NOT PER-LAYER. Older comments (and the ``raw_whiten_nlayers``
    config name, the ``_perlayer`` attribute name, and ``METHOD_TAG``) call these blocks
    "L4/L5/L6/L7". That is wrong: ``features/wavlm.py::pool_layers`` already collapses
    the selected layers with a cross-layer **mean** before the feature vector is built,
    so by the time it reaches here the layer axis is gone. The 2048-d SSL vector is
    ``[layer-averaged mean-stat (1024) | layer-averaged std-stat (1024)]``, and the four
    256-d blocks are simply contiguous **feature-axis quarters** of the layer-averaged
    mean-half — quarters of WavLM's hidden dimension, not individual layers. The
    ``_perlayer`` / ``nlayers`` names are kept only because ``method.py`` serializes and
    reloads them by those exact keys; the computation is unchanged and correct as written.
    """

    def __init__(self) -> None:
        # (n_blocks, block_dim, [(block_mean, block_whitener), ...]).  Name is legacy —
        # the blocks are feature-axis quarters, not layers — but method.py checkpoints
        # this attribute under this exact name, so it must not be renamed.
        self._perlayer: tuple[int, int, list[tuple[np.ndarray, np.ndarray]]] | None = None
        self._raw_mu: np.ndarray | None = None
        self._raw_Wz: np.ndarray | None = None

    @property
    def is_fit(self) -> bool:
        return self._raw_Wz is not None

    def fit(self, x_all: np.ndarray) -> RawWhitenTransfer:
        """Fit block whiteners + the 1024->512 PCA on the train features (in place).

        ``x_all`` is the full ``[N, 2133]`` feature matrix; only the leading 1024 columns
        (the layer-averaged mean-half) are used.
        """
        X = np.asarray(x_all, dtype=np.float64)[:, :MODEL.raw_whiten_input_dim]  # [N, 1024]
        nb = MODEL.raw_whiten_nlayers            # 4 blocks (see class docstring: these
        #                                          are feature-axis quarters, NOT layers)
        db = MODEL.raw_whiten_input_dim // nb                                    # 256
        eps = MODEL.raw_whiten_eps
        n = X.shape[0]
        scale = np.sqrt(max(n - 1, 1))

        # Both SVDs below are economy-sized, so they return only min(N, D) right
        # singular vectors.  With too few rows the block whiteners come out narrower
        # than db and the PCA basis narrower than raw_whiten_compress, and `apply`
        # then silently yields fewer than raw_whiten_compress columns -- which makes
        # Method.embed return something other than EMB_DIM and surfaces much later as
        # an opaque "input width ... matches none of ..." from score_openset.
        # Fail here, where the cause is still visible.  Shape-only check: the widths
        # depend on N alone, so this never rejects a fit that would have been correct.
        min_rows = max(db, MODEL.raw_whiten_compress)
        if n < min_rows:
            raise ValueError(
                f"RawWhitenTransfer.fit needs at least {min_rows} rows to reach the "
                f"configured widths (block {db}, PCA {MODEL.raw_whiten_compress}), got "
                f"{n}. Fitting on fewer would produce a "
                f"{MODEL.base_emb_dim + min(n, MODEL.raw_whiten_compress)}-d embedding "
                f"instead of EMB_DIM={MODEL.emb_dim}."
            )

        fits: list[tuple[np.ndarray, np.ndarray]] = []
        whitened: list[np.ndarray] = []
        for i in range(nb):
            blk = X[:, i * db:(i + 1) * db]
            mu = blk.mean(0)
            c = blk - mu
            # economy SVD whiten: columns of V scaled by 1/singular-value (unit variance).
            _, s, vt = np.linalg.svd(c, full_matrices=False)
            w = (vt.T / (s / scale + eps))                                       # [db, r]
            fits.append((mu, w))
            whitened.append(c @ w)
        Z = np.concatenate(whitened, axis=1)                                     # [N, 1024]

        # PCA-compress to 512-d (no second whitening).
        raw_mu = Z.mean(0)
        Zc = Z - raw_mu
        _, _, vt = np.linalg.svd(Zc, full_matrices=False)
        Wz = vt[:MODEL.raw_whiten_compress].T                                    # [1024, 512]

        self._perlayer = (nb, db, fits)
        self._raw_mu = raw_mu
        self._raw_Wz = Wz
        return self

    def apply(self, x_mean_half: np.ndarray) -> np.ndarray:
        """``[N, 1024]`` mean-half -> ``[N, 512]`` using the frozen train statistics.

        Caller must pass the already-sliced mean-half (``x[:, :1024]``), not the full
        2133-d feature row.
        """
        if not self.is_fit or self._perlayer is None:
            raise RuntimeError("RawWhitenTransfer.apply before fit")
        nb, db, fits = self._perlayer
        X = np.asarray(x_mean_half, dtype=np.float64)
        blocks = [((X[:, i * db:(i + 1) * db] - mu) @ w) for i, (mu, w) in enumerate(fits)]
        Z = np.concatenate(blocks, axis=1)
        return (Z - self._raw_mu) @ self._raw_Wz
