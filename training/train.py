"""
NanoPitch Training Script
=========================

This script trains the NanoPitch model to track pitch and detect voice
activity in noisy conditions. Here's what happens during training:

1. LOAD DATA: Pre-extracted mel spectrograms of clean vocals (with
   ground-truth pitch from RMVPE) and environmental noise.

2. AUGMENT: Implemented in ``augment_mel_batch`` (a stub you must fill in).
   The intended behavior is to mix each clean vocal window with a random
   noise window at a random SNR so the model sees noisy conditions during
   training. Until you implement it, training uses clean mel only.

3. PREDICT: Feed the noisy mel spectrogram to the model. It outputs
   VAD probabilities and a pitch posteriorgram.

4. COMPUTE LOSS: Compare predictions against ground truth:
   - VAD loss: Binary Cross-Entropy (is voice present? yes/no)
   - Pitch loss: BCE on the 360-dim posteriorgram, weighted by VAD
     (we only care about pitch accuracy when someone is singing)

5. BACKPROPAGATE: Compute gradients and update model weights.

6. EVALUATE: Every 5 epochs, test on held-out clips at specific SNR
   levels (-5, 0, 5, 10, 20 dB, and clean) to track progress.

Usage:
    # Train on CPU (laptop-friendly):
    python train.py --data-dir ../data --output-dir ./runs/exp1

    # Train on GPU (faster):
    python train.py --data-dir ../data --output-dir ./runs/exp1 --device cuda

    # Resume from a checkpoint:
    python train.py --data-dir ../data --output-dir ./runs/exp1 --resume ./runs/exp1/checkpoints/epoch_010.pth

    # Monitor training with TensorBoard:
    tensorboard --logdir ./runs/exp1/tb
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import (NanoPitch, f0_to_posteriorgram, PITCH_BINS, N_MELS,
                   PITCH_FMIN, PITCH_CENTS_PER_BIN)


# ═══════════════════════════════════════════════════════════════════════
# Command-Line Arguments
#
# These let you experiment with different settings without
# modifying the code. Try changing --batch-size, --lr, --gru-size, etc.
# ═══════════════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="NanoPitch trainer")

# Paths
parser.add_argument("--data-dir", type=str, default="../data",
                    help="folder containing clean.npz, noise.npz, test.npz")
parser.add_argument("--output-dir", type=str, default="./runs/default",
                    help="where to save checkpoints and TensorBoard logs")
parser.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint to resume training from")

# Device: train on CPU (laptop) or GPU (if available)
parser.add_argument("--device", type=str, default="auto",
                    help="cpu, cuda, mps, or auto (picks best available)")

# Model architecture — try changing these to see the effect on quality vs speed!
parser.add_argument("--cond-size", type=int, default=64,
                    help="conv layer width (bigger = more capacity, slower)")
parser.add_argument("--gru-size", type=int, default=96,
                    help="GRU hidden size (bigger = more memory, slower)")

# Training hyperparameters
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=32,
                    help="samples per gradient update (lower if running out of RAM)")
parser.add_argument("--lr", type=float, default=1e-3,
                    help="initial learning rate")
parser.add_argument("--seq-len", type=int, default=200,
                    help="training clip length in frames (200 = 2 seconds)")
parser.add_argument("--num-workers", type=int, default=0,
                    help="data loading threads (0 = main thread only)")
parser.add_argument("--scheduler", type=str, default="constant",
                    choices=["constant", "cosine_warmup", "cosine_warmrestart"],
                    help="LR schedule. 'constant' = upstream baseline (no decay). "
                        "'cosine_warmup' = 1-epoch LinearLR warmup + CosineAnnealingLR (per-iter). "
                        "'cosine_warmrestart' = CosineAnnealingWarmRestarts (per-epoch, T_0=--lr-t0).")
parser.add_argument("--lr-t0", type=int, default=10,
                    help="length of the first cosine cycle before the first restart")

# Data augmentation (used by augment_mel_batch once implemented)
parser.add_argument("--snr-range", type=float, nargs=2, default=[-5.0, 20.0],
                    help="min/max SNR in dB for noise mixing (see augment_mel_batch)")
parser.add_argument("--augment", type=str, default="none",
                    choices=["none", "noise", "noise_specaug"],
                    help="Input-side augmentation. 'none' = upstream baseline "
                        "(no noise mix, no SpecAugment). 'noise' = noise mixing only "
                        "(augment_mel_batch, honours --snr-range/--p-clean/--snr-bias). "
                        "'noise_specaug' = noise mixing + SpecAugment.")
# SpecAugment hyperparameters (only used when --augment=noise_specaug)
parser.add_argument("--freq-mask-param", type=int, default=4,
                    help="SpecAugment F: max width (bins) of each frequency mask.")
parser.add_argument("--n-freq-masks", type=int, default=2,
                    help="SpecAugment: number of frequency masks per sample.")
parser.add_argument("--time-mask-param", type=int, default=10,
                    help="SpecAugment T: max width (frames) of each time mask.")
parser.add_argument("--n-time-masks", type=int, default=2,
                    help="SpecAugment: number of time masks per sample.")
parser.add_argument("--p-clean", type=float, default=0.0,
                    help="probability of passing a batch row through with no noise "
                        "(0 = always mix, 0.3 = ~30%% of rows are pure clean)")
parser.add_argument("--snr-bias", type=float, default=1.0,
                    help="exponent skewing the SNR draw within snr-range; "
                        "1.0 = uniform, <1.0 = biased toward high SNR (cleaner), "
                        ">1.0 = biased toward low SNR (noisier). Try 0.5.")

# Loss weights — adjust to prioritize VAD vs pitch accuracy
parser.add_argument("--w-vad", type=float, default=0.1,
                    help="weight for VAD loss")
parser.add_argument("--w-pitch", type=float, default=1.0,
                    help="weight for pitch loss")

# VAD imbalance handling (voiced frames often <<50% of a batch)
parser.add_argument("--vad-loss", type=str, default="bce",
                    choices=["bce", "pos_weight", "focal"],
                    help="VAD loss formulation. 'bce' = uniform BCE (baseline). "
                        "'pos_weight' = upweight voiced frames by --vad-pos-weight. "
                        "'focal' = focal BCE with --vad-focal-gamma plus --vad-pos-weight.")
parser.add_argument("--vad-pos-weight", type=float, default=2.3,
                    help="weight applied to voiced-frame loss when --vad-loss is "
                        "'pos_weight' or 'focal'. ~= (1-p_voiced)/p_voiced.")
parser.add_argument("--vad-focal-gamma", type=float, default=2.0,
                    help="focusing exponent for --vad-loss=focal. Higher = more "
                        "aggressive down-weighting of easy examples. 0 = plain BCE.")

# Dataset sampling
parser.add_argument("--balanced-sampling", action="store_true",
                    help="sample clips with probability proportional to voiced-frame "
                        "fraction (clamped ≥0.05). Default: uniform random.")

# GPU-path toggles (for verifying vectorized fast paths don't move scores)
parser.add_argument("--pitch-target-build", type=str, default="gpu_vec",
                    choices=["gpu_vec", "cpu_loop"],
                    help="How to build the pitch posteriorgram target each batch. "
                        "'gpu_vec' = vectorized on-device (default, fast). "
                        "'cpu_loop' = per-sample numpy loop via f0_to_posteriorgram "
                        "To validate against baseline")

parser.add_argument("--pitch-sigma", type=float, default=1.2,
                    help="Gaussian sigma (in bins) for pitch posteriorgram target. "
                        "Lower = sharper supervision (try 0.8).")
parser.add_argument("--pitch-mask", type=str, default="vad",
                    choices=["vad", "strict"],
                    help="Mask used for pitch loss. 'vad' = clean_vad (broad), "
                        "'strict' = (f0>0) — recommended to avoid training pitch head "
                        "on silent frames marked voiced by clean_vad.")
#Label mismatch
parser.add_argument("--vad-target", type=str, default="f0",
                    choices=["f0", "clean_vad"],
                    help="VAD training label. 'f0' = (clean_f0 > 0), upstream baseline. "
                        "'clean_vad' = dataset clean_vad array (broader).")
parser.add_argument("--eval-vdr", type=str, default="pitch",
                    choices=["pitch", "vad"],
                    help="Source of voicing call for eval VDR metric. "
                        "'pitch' = pitch-decode (f0_decoded > 0), upstream baseline. "
                        "'vad' = VAD head (pv > 0.5).")

# Early stopping patience (epochs to wait for macro RPA improvement before stopping)
parser.add_argument("--patience", type=int, default=0,
                    help="stop if macro-avg RPA does not improve for this many epochs "
                        "(0 = disabled). Use >= 20 to survive LR warm restarts)")

# CLI flags for curriculum learning
parser.add_argument("--curriculum", action="store_true",
                    help="staged head training: VAD-only → pitch-only → joint fine-tune")
parser.add_argument("--curriculum-vad-epochs", type=int, default=10,
                    help="epochs for VAD-only phase (Phase 1)")
parser.add_argument("--curriculum-pitch-epochs", type=int, default=20,
                    help="epochs for pitch-only phase (Phase 2)")
# ═══════════════════════════════════════════════════════════════════════
# Dataset
#
# Loads pre-extracted features from .npz files and creates training
# examples by pairing clean vocal segments with noise segments.
# ═══════════════════════════════════════════════════════════════════════

class NanoPitchDataset(Dataset):
    """PyTorch Dataset that serves (clean_mel, noise_mel, vad, pitch) tuples.

    Noise mixing is applied in ``augment_mel_batch`` in the training loop,
    not here — so each epoch can see different random mixtures once that
    function is implemented.
    """

    def __init__(self, data_dir, seq_len=200, vad_target="f0",
                 balanced_sampling=False):
        self.seq_len = seq_len
        self.vad_target = vad_target
        self.balanced_sampling = balanced_sampling

        # Load pre-extracted features (stored as float16 to save disk space)
        print("Loading clean.npz...")
        clean = np.load(os.path.join(data_dir, "clean.npz"))
        self.clean_mel = clean["mel"]        # (total_frames, 40) — mel spectrogram
        self.clean_f0 = clean["f0"]          # (total_frames,) — RMVPE f0 in Hz
        self.clean_vad = clean["vad"]        # (total_frames,) — voice activity
        self.clean_lengths = clean["lengths"] # length of each clip

        print("Loading noise.npz...")
        noise = np.load(os.path.join(data_dir, "noise.npz"))
        self.noise_mel = noise["mel"]         # (total_frames, 40)
        self.noise_lengths = noise["lengths"]

        # Build a list of (start, end) indices for clips long enough to sample from
        self.clean_segments = self._build_segments(self.clean_lengths, seq_len)
        self.noise_segments = self._build_segments(self.noise_lengths, seq_len)

        print(f"  Clean: {len(self.clean_mel):,} frames, "
            f"{len(self.clean_segments)} usable segments")
        print(f"  Noise: {len(self.noise_mel):,} frames, "
            f"{len(self.noise_segments)} usable segments")

        self.rng = np.random.default_rng()
        if self.balanced_sampling:
            # Voiced-fraction weights — clip probability ∝ voiced fraction
            # (clamped ≥0.05 so near-silent clips still draw occasionally).
            self.voiced_weights = np.array([
                max(float(np.mean(self.clean_vad[s:e])), 0.05)
                for s, e in self.clean_segments],
                dtype=np.float64)
            self.voiced_weights /= self.voiced_weights.sum()
        else:
            self.voiced_weights = None

    def _build_segments(self, lengths, min_len):
        """Find all contiguous segments at least min_len frames long."""
        segments = []
        offset = 0
        for length in lengths:
            if length >= min_len:
                segments.append((offset, offset + length))
            offset += length
        return segments

    def __len__(self):
        # Return a reasonable epoch size (not too long for CPU training)
        return min(len(self.clean_segments) * 3, 10000)

    def __getitem__(self, idx):
        # Pick a random clean segment and extract a random window
        if self.voiced_weights is not None:
            seg_idx = self.rng.choice(len(self.clean_segments), p=self.voiced_weights)
        else:
            seg_idx = self.rng.integers(len(self.clean_segments))
        start, end = self.clean_segments[seg_idx]
        offset = self.rng.integers(0, end - start - self.seq_len + 1)
        s = start + offset

        mel_clean = self.clean_mel[s:s + self.seq_len].astype(np.float32)
        f0 = self.clean_f0[s:s + self.seq_len].astype(np.float32)
        if self.vad_target == "clean_vad":
            vad = self.clean_vad[s:s + self.seq_len].astype(np.float32)
        else:  # "f0" — upstream baseline
            vad = (self.clean_f0[s:s + self.seq_len] > 0).astype(np.float32)

        # Pick a random noise segment (independently)
        noise_idx = self.rng.integers(len(self.noise_segments))
        ns, ne = self.noise_segments[noise_idx]
        n_offset = self.rng.integers(0, ne - ns - self.seq_len + 1)
        ns = ns + n_offset
        mel_noise = self.noise_mel[ns:ns + self.seq_len].astype(np.float32)

        return mel_clean, mel_noise, vad, f0


def augment_mel_batch(mel_clean, mel_noise, snr_range, device,
                      p_clean=0.0, snr_bias=1.0):
    """Training-time augmentation: mix clean and noise log-mel.

    Parameters
    ----------
    mel_clean, mel_noise : Tensor, shape (B, T, N_MELS), on ``device``
    snr_range : (float, float) — min and max SNR in dB (see ``--snr-range``)
    device : torch.device

    Returns
    -------
    Tensor (B, T, N_MELS) — mel passed to the model.

    Student exercise
    ----------------
    Implement random SNR mixing. For each batch row, draw ``snr_db`` uniformly
    from ``snr_range``, convert to a log-domain gain, and combine clean and
    noise with ``torch.logaddexp`` for numerical stability, e.g.::

        B = mel_clean.size(0)
        snr_db = (torch.rand(B, 1, 1, device=device)
                  * (snr_range[1] - snr_range[0]) + snr_range[0])
        gain_offset = -snr_db * (np.log(10.0) / 20.0)
        return torch.logaddexp(mel_clean, mel_noise + gain_offset)

    This stub returns ``mel_clean`` unchanged so the trainer runs without
    augmentation until you add the above (or your own variant).
    """
    # return mel_clean
    # ── Previous: uniform SNR draw, every row is mixed with noise ──
    # B = mel_clean.size(0)
    # snr_db = (torch.rand(B, 1, 1, device=device)
    #         * (snr_range[1] - snr_range[0]) + snr_range[0])
    # gain_offset = -snr_db * (np.log(10.0) / 20.0)
    # return torch.logaddexp(mel_clean, mel_noise + gain_offset)

    # ── New: bias toward cleaner end ──
    # snr_bias skews the uniform draw: u**snr_bias with bias<1 pushes mass
    # toward u=1 (high SNR); bias=1.0 keeps uniform behavior.
    # p_clean randomly passes a fraction of batch rows through as pure clean
    # (no noise), restoring some of the clean-side calibration the model loses
    # when every training example is noise-augmented.
    B = mel_clean.size(0)
    u = torch.rand(B, 1, 1, device=device)
    if snr_bias != 1.0:
        u = u.pow(snr_bias)
    snr_db = u * (snr_range[1] - snr_range[0]) + snr_range[0]
    gain_offset = -snr_db * (np.log(10.0) / 20.0)
    mixed = torch.logaddexp(mel_clean, mel_noise + gain_offset)
    if p_clean > 0.0:
        clean_mask = torch.rand(B, 1, 1, device=device) < p_clean
        mixed = torch.where(clean_mask, mel_clean, mixed)
    return mixed

# SpecAugment (Frequency and time masking)
def spec_augment(mel, n_freq_masks=2, freq_mask_param=4,
                n_time_masks=2, time_mask_param=10):
    """SpecAugment: per-sample random frequency and time masking.

    Each sample in the batch gets independent random masks so the model
    cannot memorise mask positions. Keep masks small (F≤4, T≤10) to
    avoid destroying the pitch information the model needs to learn from.
    """
    mel = mel.clone()
    B, seq_len, n_mels = mel.shape
    for b in range(B):
        for _ in range(n_freq_masks):
            f = torch.randint(1, freq_mask_param + 1, (1,)).item()
            f0 = torch.randint(0, max(n_mels - f, 1), (1,)).item()
            mel[b, :, f0:f0 + f] = 0.0
        for _ in range(n_time_masks):
            t = torch.randint(1, time_mask_param + 1, (1,)).item()
            t0 = torch.randint(0, max(seq_len - t, 1), (1,)).item()
            mel[b, t0:t0 + t, :] = 0.0
    return mel

def _macro_rpa(eval_results):
    """Macro-average RPA across conditions — the leaderboard metric."""
    all_rpa = [v['rpa'] for v in eval_results.values()
               if not np.isnan(v.get('rpa', float('nan')))]
    return float(np.mean(all_rpa)) if all_rpa else float('nan')

def _set_phase(model, phase):
    """Toggle requires_grad for curriculum training phases.

    AdamW respects requires_grad at step() time — frozen parameters receive
    no gradient and are not updated, but their Adam momentum state is preserved
    so they resume correctly when unfrozen in Phase 3.
    """
    for name, p in model.named_parameters():
        if phase == 'vad':
            p.requires_grad = ('dense_pitch' not in name)
        elif phase == 'pitch':
            p.requires_grad = ('dense_vad' not in name)
        else:  # joint
            p.requires_grad = True

# ═══════════════════════════════════════════════════════════════════════
# Training Loop (one epoch)
# ═══════════════════════════════════════════════════════════════════════

def _vad_loss(pred, target, args, bce_none):
    """Dispatch VAD loss: plain BCE, pos-weighted BCE, or focal BCE.

    pred/target are probabilities (model applies sigmoid); bce_none is an
    instance of nn.BCELoss(reduction='none').
    """
    per_elem = bce_none(pred, target)
    if args.vad_loss == "bce":
        return per_elem.mean()
    # Upweight voiced frames by pos_weight; unvoiced keep weight 1.
    pw = torch.where(target > 0.5,
                    torch.full_like(target, args.vad_pos_weight),
                    torch.ones_like(target))
    if args.vad_loss == "pos_weight":
        return (pw * per_elem).mean()
    # focal: (1 - p_t)^gamma modulator on pos-weighted BCE
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    modulator = (1.0 - pt).clamp(min=0.0).pow(args.vad_focal_gamma)
    return (pw * modulator * per_elem).mean()


def train_one_epoch(model, dataloader, optimizer, scheduler, writer,
                    epoch, device, args, global_step_offset=0):
    model.train()  # enable dropout, batch norm, etc. (if any)
    bce = nn.BCELoss(reduction='none')  # per-element BCE, we'll weight manually
    running = {'loss': 0, 'vad': 0, 'pitch': 0}
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", unit="batch")
    for batch_idx, (mel_clean, mel_noise, vad_target, f0_target) in enumerate(pbar):
        # Move data to the training device (CPU or GPU)
        mel_clean = mel_clean.to(device)
        mel_noise = mel_noise.to(device)
        vad_target = vad_target.to(device)

        # Build pitch posteriorgram target on-the-fly from f0 values.
        # This saves huge amounts of RAM (f0 is 1 float vs 360 for full posterior).
        f0_dev = f0_target.to(device)                                              # (B, T)
        if args.pitch_target_build == "cpu_loop":
            # Upstream reference: per-sample numpy loop via f0_to_posteriorgram.
            B, T = f0_target.shape
            pitch_target = torch.zeros(B, T, PITCH_BINS, device=device)
            f0_np = f0_target.cpu().numpy()
            for b in range(B):
                pg = f0_to_posteriorgram(f0_np[b], n_frames=T,
                                         sigma_bins=args.pitch_sigma)
                pitch_target[b] = torch.from_numpy(pg).to(device)
        else:
            # Vectorized on-device. Mirrors f0_to_posteriorgram (Gaussian bump of
            # sigma bins centered on each voiced frame's bin index). Unvoiced
            # frames get all zeros.
            voiced_tgt = (f0_dev > 0).float().unsqueeze(-1)                        # (B, T, 1)
            safe_f0 = torch.where(f0_dev > 0, f0_dev, torch.ones_like(f0_dev))
            bins = 1200.0 * torch.log2(safe_f0 / PITCH_FMIN) / PITCH_CENTS_PER_BIN # (B, T)
            bin_grid = torch.arange(PITCH_BINS, device=device,
                                    dtype=torch.float32).view(1, 1, -1)            # (1, 1, 360)
            dist = bin_grid - bins.unsqueeze(-1).float()                           # (B, T, 360)
            pitch_target = torch.exp(-0.5 * (dist / args.pitch_sigma) ** 2) * voiced_tgt

        # ── Data augmentation (flag-gated for baseline compatibility) ──
        if args.augment == "none":
            mel_mix = mel_clean                              # upstream baseline
        elif args.augment == "noise":
            mel_mix = augment_mel_batch(mel_clean, mel_noise, args.snr_range, device,
                                        p_clean=args.p_clean, snr_bias=args.snr_bias)
        else:  # "noise_specaug"
            mel_mix = augment_mel_batch(mel_clean, mel_noise, args.snr_range, device,
                                        p_clean=args.p_clean, snr_bias=args.snr_bias)
            mel_mix = spec_augment(mel_mix,
                                   n_freq_masks=args.n_freq_masks,
                                   freq_mask_param=args.freq_mask_param,
                                   n_time_masks=args.n_time_masks,
                                   time_mask_param=args.time_mask_param)

        # ── Forward Pass ──
        # Causal convs → output same length as input, no trimming needed.
        pred_vad, pred_pitch, _ = model(mel_mix)

        # ── Loss Computation ──
        vad_loss = _vad_loss(pred_vad.squeeze(-1), vad_target, args, bce)

        # Pitch loss: BCE on the 360-dim posteriorgram, but weighted
        # by VAD — we don't penalize pitch errors on silent frames
        if args.pitch_mask == "strict":
            voiced_weight = (f0_dev > 0).float().unsqueeze(-1)  # matches pitch_target
        else:
            voiced_weight = vad_target.unsqueeze(-1)  # broad clean_vad
        pitch_loss = (voiced_weight * bce(pred_pitch, pitch_target)).mean()

        loss = args.w_vad * vad_loss + args.w_pitch * pitch_loss

        # ── Backward Pass ──
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        writer.add_scalar("train/grad_norm", grad_norm, global_step_offset + batch_idx)
        if getattr(args, "_scheduler_step", "epoch") == "iter":
            scheduler.step()   # per-iteration step (cosine_warmup)

        # ── Logging ──
        running['loss'] += loss.item()
        running['vad'] += vad_loss.item()
        running['pitch'] += pitch_loss.item()
        n_batches += 1

        pbar.set_postfix(
            loss=f"{running['loss']/n_batches:.4f}",
            vad=f"{running['vad']/n_batches:.4f}",
            pitch=f"{running['pitch']/n_batches:.4f}",
        )

    if n_batches == 0:
        warnings.warn(
            "No batches were processed in this epoch (likely due to drop_last=True "
            "with an oversized batch-size). Returning NaN loss for this epoch.",
            RuntimeWarning,
        )
        writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
        return float("nan")

    # Log to TensorBoard (view with: tensorboard --logdir ./runs/exp1/tb)
    for key in running:
        writer.add_scalar(f"train/{key}", running[key] / n_batches, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], epoch)
    return running['loss'] / n_batches


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
#
# Tests the model on held-out clips at specific noise levels.
# This shows how well the model performs as conditions get harder.
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()  # no gradients needed during evaluation (saves memory)
def evaluate(model, data_dir, writer, epoch, device, args):
    """Evaluate pitch tracking and VAD at each noise level."""
    from model import viterbi_decode
    warnings.warn(
        "train.py evaluation uses offline Viterbi only and a reduced metric set. "
        "Use training/evaluate.py for browser-matching realtime metrics.",
        RuntimeWarning,
    )
    model.eval()  # disable dropout etc.

    test_path = os.path.join(data_dir, "test.npz")
    if not os.path.exists(test_path):
        print("  [eval] test.npz not found, skipping")
        return

    test = np.load(test_path)
    clips = test['clips']       # (N, T, 40) — noisy mel input
    f0_all = test['f0']         # (N, T) — ground-truth f0 in Hz
    vad_all = test['vad']       # (N, T) — ground-truth VAD
    snrs = test['snr']          # (N,) — SNR of each clip
    N = clips.shape[0]

    # Evaluate each clip
    clip_results = []
    for i in range(N):
        mel = torch.from_numpy(clips[i].astype(np.float32)).unsqueeze(0).to(device)
        v, p, _ = model(mel)
        pv = v.squeeze(0).cpu().numpy().squeeze(-1)
        pp = p.squeeze(0).cpu().numpy()
        T = pv.shape[0]

        vr = vad_all[i, :T].astype(np.float32)
        f0r = f0_all[i, :T].astype(np.float32)

        # Decode predicted posteriorgram to f0
        f0d = viterbi_decode(pp)

        # VAD accuracy: fraction of frames with correct voice/silence label
        vacc = float(np.mean((pv > 0.5) == (vr > 0.5)))

        # Voicing detection rate: of the frames that ARE voiced, how many
        # did the model correctly identify as voiced?
        vg = f0r > 0
        if getattr(args, "eval_vdr", "pitch") == "vad":
            vp = pv > 0.5                      # VAD head voicing call
        else:
            vp = f0d > 0                       # pitch-decode voicing (baseline)
        vdr = float(np.mean(vp[vg])) if vg.sum() > 0 else np.nan

        # Raw Pitch Accuracy: of frames where both ground-truth and
        # prediction are voiced, what fraction have pitch error < 50 cents?
        both = vg & vp
        if both.sum() > 0:
            ce = np.abs(1200 * np.log2(f0d[both] / (f0r[both] + 1e-10) + 1e-10))
            rpa = float(np.mean(ce < 50))
        else:
            rpa = np.nan
        clip_results.append({'snr': float(snrs[i]), 'vad_acc': vacc,
                             'vdr': vdr, 'rpa': rpa})

    # Group results by SNR level and print a summary table
    by_snr = {}
    for r in clip_results:
        by_snr.setdefault(r['snr'], []).append(r)

    def smean(vals):
        vals = [v for v in vals if not np.isnan(v)]
        return np.mean(vals) if vals else float('nan')

    results = {}
    print(f"\n  {'Condition':<10s}  {'VAD Acc':>8s}  {'VDR':>8s}  {'RPA':>8s}")
    print(f"  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*8}")

    for snr in sorted(by_snr.keys(), key=lambda x: x if np.isfinite(x) else 1e6):
        c = by_snr[snr]
        tag = "clean" if not np.isfinite(snr) else f"{snr:+.0f} dB"
        va = smean([x['vad_acc'] for x in c])
        vd = smean([x['vdr'] for x in c])
        rp = smean([x['rpa'] for x in c])
        print(f"  {tag:<10s}  {va:8.1%}  {vd:8.1%}  {rp:8.1%}")
        results[tag] = {'vad_acc': va, 'vdr': vd, 'rpa': rp}

        if writer:
            stag = tag.replace(' ', '').replace('+', 'p').replace('-', 'n')
            writer.add_scalar(f"eval/vad_acc_{stag}", va, epoch)
            writer.add_scalar(f"eval/vdr_{stag}", vd, epoch)
            writer.add_scalar(f"eval/rpa_{stag}", rp, epoch)
    print()
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main — ties everything together
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parser.parse_args()

    # Auto-detect the best available device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")  # Apple Silicon
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    data_dir = os.path.abspath(args.data_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create model
    model = NanoPitch(cond_size=args.cond_size, gru_size=args.gru_size)
    start_epoch = 1

    # Optionally resume from a checkpoint
    resume_ckpt = None
    if args.resume:
        warnings.warn("Loading checkpoint via torch.load() executes Python deserialization. "
                      "Only use checkpoints from trusted sources.", RuntimeWarning)
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(resume_ckpt["state_dict"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        print(f"Resumed from epoch {start_epoch - 1}")

    model.to(device)

    # Create data pipeline
    dataset = NanoPitchDataset(data_dir, seq_len=args.seq_len,
                               vad_target=args.vad_target,
                               balanced_sampling=args.balanced_sampling)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True,
                            pin_memory=(device.type == "cuda"),
                            persistent_workers=(args.num_workers > 0))

    # AdamW optimizer — Adam with weight decay (a form of regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.8, 0.98), eps=1e-8)

    # Learning rate scheduler — student exercise.
    from torch.optim.lr_scheduler import (LinearLR, CosineAnnealingLR,
                                           SequentialLR, LambdaLR,
                                           CosineAnnealingWarmRestarts)
    if args.scheduler == "constant":
        # Upstream baseline: constant LR (stepped per-epoch, no-op).
        scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        scheduler_step = "epoch"
    elif args.scheduler == "cosine_warmup":
        total_iters = len(dataloader) * args.epochs
        warmup_iters = len(dataloader)  # 1 epoch of warmup
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
        cosine = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters,
                                   eta_min=1e-5)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[warmup_iters])
        scheduler_step = "iter"
    elif args.scheduler == "cosine_warmrestart":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.lr_t0,
                                                T_mult=2, eta_min=1e-5)
        scheduler_step = "epoch"
    args._scheduler_step = scheduler_step
    
    # load checkpoint states into optimizer and scheduler if resuming
    if resume_ckpt:
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            print("Optimizer state loaded from checkpoint.")
        if "scheduler" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler"])
            print("Scheduler state loaded from checkpoint.")
        del resume_ckpt  # free up memory

    # TensorBoard writer for visualizing training progress
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb"))

    # ── Training loop ──
    best_loss = float("inf")
    best_macro_rpa = 0.0      # unweighted macro RPA → equivalent to best macro gross-err
    patience_counter = 0
    global_step_offset = 0
    
    # Set requires_grad according to curriculum learning phase (if enabled)
    if args.curriculum:
        vad_phase_end   = start_epoch + args.curriculum_vad_epochs
        pitch_phase_end = vad_phase_end + args.curriculum_pitch_epochs
        print(f"Curriculum: VAD-only [{start_epoch},{vad_phase_end}), "
            f"pitch-only [{vad_phase_end},{pitch_phase_end}), "
            f"joint [{pitch_phase_end},{start_epoch + args.epochs})")
    else:
        vad_phase_end = pitch_phase_end = start_epoch - 1  # never trigger

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()
        # Update curriculum phase if enabled
        if args.curriculum:
            if epoch == start_epoch:
                _set_phase(model, 'vad')
                print(f"Starting Phase 1: VAD-only training for {args.curriculum_vad_epochs} epochs.")
            elif epoch == vad_phase_end:
                _set_phase(model, 'pitch')
                print(f"Starting Phase 2: pitch-only training for {args.curriculum_pitch_epochs} epochs.")
            elif epoch == pitch_phase_end:
                _set_phase(model, 'joint')
                # Joint phase: pure cosine decay (no warm restarts) so a mid-phase
                # LR spike can't knock the pitch head off its minimum.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * 0.2  # reduce LR for fine-tuning phase
                joint_epochs = start_epoch + args.epochs - pitch_phase_end
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=joint_epochs, eta_min=1e-5)
                # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                #     optimizer, T_0=args.lr_t0, T_mult=2, eta_min=1e-5)
                print(f"Starting Phase 3: joint fine-tuning for the remaining "
                      f"{joint_epochs} epochs")
        train_loss = train_one_epoch(model, dataloader, optimizer, scheduler,
                                    writer, epoch, device, args,
                                    global_step_offset=global_step_offset)
        global_step_offset += len(dataloader)  # keep track of total steps for LR scheduling
        if args._scheduler_step == "epoch":
            scheduler.step()   # per-epoch step (constant / cosine_warmrestart)
        # else: per-iter step already happened inside train_one_epoch
        dt = time.time() - t0
        print(f"  Epoch {epoch} done in {dt:.1f}s, loss={train_loss:.5f}")

        # Save checkpoint after every epoch
        ckpt = {"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "model_kwargs": {"cond_size": args.cond_size,
                                "gru_size": args.gru_size},
                "args": vars(args),
                "loss": train_loss}
        torch.save(ckpt, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
        
        # Evaluate every 5 epochs (and on the first epoch)
        if epoch % 5 == 0 or epoch == start_epoch:
            eval_results = evaluate(model, data_dir, writer, epoch, device, args)
            if eval_results:
                rpa_macro = _macro_rpa(eval_results)
                print(f"  RPA  macro={rpa_macro:.4f}")
                # Track best macro RPA — this is the leaderboard metric.
                # On the offline decoder, macro gross_err = 1 - macro_rpa exactly,
                # so this is also the best-macro-gross-err checkpoint at zero extra cost.
                if not np.isnan(rpa_macro) and rpa_macro > best_macro_rpa:
                    best_macro_rpa = rpa_macro
                    patience_counter = 0
                    torch.save(ckpt, os.path.join(ckpt_dir, "best_macro_rpa.pth"))
                    print(f" New best macro RPA: {best_macro_rpa:.4f} "
                        f"(macro gross_err={1.0 - best_macro_rpa:.4f}) — checkpoint saved.")
                elif not np.isnan(rpa_macro):
                    patience_counter += 5
                    if args.patience > 0 and patience_counter >= args.patience:
                        print(f" Early stopping triggered after {epoch} epochs without improvement "
                            f"(best macro RPA: {best_macro_rpa:.4f}).")
                        break

        # Also save as "best" if this is the lowest loss so far
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pth"))

    writer.close()
    print(f"\nTraining complete. Best loss: {best_loss:.5f}")
    print(f"Best loss       checkpoint : {ckpt_dir}/best.pth")
    print(f"Best macro-RPA  checkpoint : {ckpt_dir}/best_macro_rpa.pth  "
          f"(macro_rpa={best_macro_rpa:.4f}, macro_gross_err={1.0 - best_macro_rpa:.4f})")
    
    #cuda cache cleanup
    import gc
    del dataset, dataloader, model, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    main()
