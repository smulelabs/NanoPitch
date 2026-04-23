# NanoPitch

> **Note:** This is a modified version of the original [NanoPitch](https://github.com/smulelabs/NanoPitch) repo adapted for classroom use. It adds a student submission and evaluation system with an auto-updating leaderboard.
>
> - 📊 **[View Leaderboard](https://charisrenee.github.io/NanoPitch-ClassLeaderboard/)** — interactive charts, sortable table (RPA, Gross Error Rate, VDR, VAD across all noise conditions)
> - 📋 **[Static Leaderboard Table](LEADERBOARD.md)** — plain Markdown version
<<<<<<< Updated upstream
> - 🤝 **[How to Submit](CONTRIBUTING.md)** — instructions for adding your model and appearing on the leaderboard
=======
> - 🤝 **[How to Submit](CONTRIBUTING.md)** — fork this repo, add your submission, and open a PR back to **this** repo (not the original NanoPitch upstream)
>>>>>>> Stashed changes

---

Train a neural network that tracks pitch and detects singing in real-time on your laptop, then deploy it to run live in a web browser.

## What You'll Learn

- **How neural networks process audio** — converting sound to mel spectrograms, feeding them through recurrent layers (GRUs), and interpreting the output
- **Data augmentation** — implement `augment_mel_batch` in `train.py` to mix clean vocals with noise at random SNR (stub returns clean-only until you do)
- **Training and evaluation** — loss functions, backpropagation, TensorBoard monitoring, and quantitative metrics (pitch accuracy, VAD accuracy)
- **Deployment** — exporting a PyTorch model to make inference run in C, then run it in a browser via WebAssembly (WASM)

## How It Works

NanoPitch takes a **mel spectrogram** (a time-frequency representation of audio) and predicts:

1. **Is someone singing right now?** (Voice Activity Detection / VAD)
2. **What pitch are they singing?** (360 per-bin sigmoid scores covering 6 octaves)

The model is small enough (~333K parameters) to train on a laptop CPU in 6-8 hours, and to run in real-time in a browser. The architecture is adapted from RNNoise \[4\].

### Architecture

```
Audio (16 kHz) -> Mel Spectrogram (40 bands, 10ms frames)

    +-- Causal Conv1d(40->64, k=3) + tanh --+
    |  Causal Conv1d(64->96, k=3) + tanh    |  Feature extraction
    +---------------------------------------+
    |  GRU (96 units) x 3 layers            |  Temporal modeling
    +---------------------------------------+
    |  Concatenate all layer outputs        |  Skip connections
    |  Dense -> sigmoid                     |
    +-------------------+-------------------+
                        |-- VAD (1 value)
                        +-- Pitch (360 sigmoid scores)
```

### Causal Convolutions

The conv layers use **causal (left) padding**: `output[t]` only depends on `input[t-2], input[t-1], input[t]` — never on future frames. This is essential for real-time streaming: when a new audio frame arrives, we can produce an output without waiting for future frames (no look-ahead latency from the conv stack).

Without causal padding, a kernel-3 conv would peek into the future (`input[t], input[t+1], input[t+2]`), which works fine during offline training but makes real-time deployment impossible.

### Data

Pre-extracted features are provided in three files:

| File | Contents | Source |
|------|----------|--------|
| `data/clean.npz` | Mel spectrograms + RMVPE pitch posteriorgrams + VAD | GTSinger \[1\] (9 languages) |
| `data/noise.npz` | Mel spectrograms of environmental noise | FSDNoisy18k \[3\] |
| `data/test.npz` | Pre-mixed test clips at 6 noise levels | Clean + noise at -5, 0, 5, 10, 20 dB, and clean |

Download these files from Hugging Face: <https://huggingface.co/datasets/smulelabs/NanoPitch-PreExtract>.

Training calls `augment_mel_batch` in `train.py`. The starter code returns clean mel only; you implement random-SNR noise mixing there (see the function docstring for a `logaddexp` recipe in the log-mel domain — equivalent to linear-domain `clean + noise`, but numerically stable).

## Getting Started

### Prerequisites

```bash
pip install torch numpy scipy tensorboard tqdm huggingface_hub
```

### 1. Download data

```bash
python scripts/download_data.py --output-dir data
```

### 2. Train

```bash
cd training
python train.py --data-dir ../data --output-dir ./runs/my_first_model

# Monitor in another terminal:
tensorboard --logdir ./runs/my_first_model/tb
```

**Things to experiment with:**
- `--gru-size 64` vs `128` — smaller/larger model
- `--snr-range -10 30` — wider SNR range once `augment_mel_batch` is implemented
- `--lr 0.0003` — different learning rate
- `--w-pitch 2.0` — prioritize pitch accuracy over VAD

### 3. Evaluate

```bash
python evaluate.py --checkpoint ./runs/my_first_model/checkpoints/best.pth --data-dir ../data
```

This prints two tables — one for each decoding strategy:

```
========================================================================
  NanoPitch — Viterbi (Offline)
========================================================================

  Condition    VAD Acc       VDR       RPA       RCA     Gross     Med.c
  ----------  --------  --------  --------  --------  --------  --------
  -5 dB          85.2%    78.3%    52.1%    55.4%    47.9%     42.3
  ...
  clean          97.5%    95.2%    89.1%    91.3%    10.9%      8.4

========================================================================
  NanoPitch — Viterbi (Realtime — matches browser)
========================================================================

  Condition    VAD Acc       VDR       RPA       RCA     Gross     Med.c
  ...
```

### 4. Deploy to Browser

```bash
# Export model weights to JSON:
cd ../deployment
python export_weights.py ../training/runs/my_first_model/checkpoints/best.pth -o web/model.json

# Build WASM (requires Emscripten SDK):
cd wasm
source /path/to/emsdk/emsdk_env.sh
./build.sh

# Serve the web app:
cd ../web
python -m http.server 8080
# Open http://localhost:8080, drag model.json onto the page, click "Start Microphone"
```

## Key Concepts

### Mel Spectrogram

A mel spectrogram converts raw audio into a 2D representation: time on one axis, frequency on the other. The "mel" part means the frequency axis is warped to match human pitch perception (we hear the difference between 200 Hz and 400 Hz as the same "distance" as 1000 Hz and 2000 Hz). We use 40 mel bands, computed every 10ms with a 25ms analysis window.

### GRU (Gated Recurrent Unit)

GRUs are recurrent neural networks designed for sequential data \[5\]. At each time step, a GRU takes the current input and its previous hidden state, and produces a new hidden state. "Gates" (learned sigmoid functions) control how much old information to keep vs. replace. This lets the network track pitch continuously across frames — it "remembers" what it heard before.

### Pitch Posteriorgram

Instead of predicting a single pitch value, the model outputs 360 independent sigmoid scores (one per pitch bin at 20-cent resolution). This covers 6 octaves from approximately B0 (31.7 Hz) to approximately B6 (~2006 Hz). The score pattern captures uncertainty: if the model isn't sure between two pitches, both bins can have moderate confidence.

### Viterbi Decoding

The raw 360-dim output needs to be decoded into a single f0 value per frame. Viterbi decoding \[6\] is a dynamic programming algorithm that finds the most likely *sequence* of pitch states, not just the best state per frame. It enforces constraints:
- Pitch can't jump more than 240 cents (2.4 semitones) per frame
- Switching between voiced and unvoiced costs a penalty

We provide two implementations:

- **Offline Viterbi**: Processes the whole sequence, then backtraces from the end to find the globally optimal path. Better accuracy but requires seeing the full sequence first. Used during evaluation as an upper bound.

- **Realtime Viterbi**: Processes one frame at a time, emitting the best state immediately. This is what runs in the browser — it can't "change its mind" about past frames when new evidence arrives. The evaluation shows both so you can see the gap.

### VAD (Voice Activity Detection)

VAD answers "is someone singing/speaking right now?" — a binary probability per frame. It's trained from RMS energy thresholding at -30 dB. During training, the pitch loss is weighted by VAD so the model only learns pitch on voiced frames (we don't care what pitch it predicts during silence).

### Data augmentation (`augment_mel_batch`)

The default stub does not mix noise; implement mixing in `training/train.py` so each training step uses noisy mel (e.g. random SNR in `--snr-range`, default -5 to +20 dB). At test time, `test.npz` still evaluates at fixed SNR levels so you can measure robustness after you add augmentation.

### Real-Time Factor (RTF)

RTF = inference_time / audio_time. Each audio frame is 10ms, so if inference takes 3ms, RTF = 0.3. The web app (WASM path) shows a live RTF sparkline and numeric meter:

- **Green (RTF < 0.5)**: Plenty of headroom, smooth real-time
- **Yellow (RTF 0.5-1.0)**: Getting close to the limit
- **Red (RTF > 1.0)**: Too slow — frames are being dropped

RTF depends on your model size and the user's hardware. The default model (`--gru-size 96`, 333K params) typically runs at RTF ~0.1-0.3 in WASM. Larger models cost more:

```
--gru-size 64  → ~200K params, RTF ~0.05-0.15
--gru-size 96  → ~333K params, RTF ~0.1-0.3  (default)
--gru-size 128 → ~500K params, RTF ~0.2-0.5
--gru-size 256 → ~1.6M params, RTF ~0.5-1.5  (may drop frames)
```

This is a key lesson in deploying ML models: accuracy and latency are in tension. A bigger model tracks pitch better, but if it can't run in real-time, it's useless for a live application. The RTF plot lets you see this tradeoff directly. Note that "no look-ahead latency" from causal convs does not mean zero total delay: the 25ms analysis window and startup warm-up still add initial delay before valid output.

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **VAD Acc** | Fraction of frames with correct voice/silence label |
| **VDR** | Voicing Detection Rate — of truly voiced frames, how many detected? |
| **RPA** | Raw Pitch Accuracy — of voiced frames, how many within 50 cents? |
| **RCA** | Raw Chroma Accuracy — same as RPA but ignoring octave errors |
| **Gross** | Gross error rate — frames with pitch error > 50 cents |
| **Med.c** | Median pitch error in cents (100 cents = 1 semitone) |

The test set evaluates at 6 conditions: -5, 0, +5, +10, +20 dB SNR, and clean (no noise). This shows how the model degrades with increasing noise.

## Debug Guide & Experiments

The baseline training script is intentionally minimal. Here are six concrete directions for improving the model, each with implementation pointers and relevant literature.

---

### 1. Log-mel augmentation: you can't just add noise linearly

After implementing `augment_mel_batch`, you may want to explore additional augmentation strategies. A critical subtlety: **log-mel features are in log-power space**, so additive arithmetic does not correspond to additive mixing of audio signals.

**Why it matters.** If you naively do `mel_mix = mel_clean + mel_noise` (linear addition in log space), you are effectively *multiplying* the spectral envelopes in the linear domain — which is physically meaningless (it models convolution of the two sources, not their acoustic sum). The correct log-domain mixing is:

```python
mel_mix = torch.logaddexp(mel_clean, mel_noise + gain_offset)
# equivalent to: log(exp(mel_clean) + exp(mel_noise * scale))
```

**Further augmentation to try.** SpecAugment \[7\] masks rectangular blocks of time steps and frequency channels directly on the log-mel input. This regularises the model against partial occlusions in the spectrogram and is simple to implement:

```python
# Frequency masking: zero out F consecutive mel bands, starting at f0
f0 = random.randint(0, N_MELS - F)
mel[:, :, f0:f0+F] = 0.0

# Time masking: zero out T consecutive frames, starting at t0
t0 = random.randint(0, seq_len - T)
mel[:, t0:t0+T, :] = 0.0
```

Start with small masks (F ≤ 8 bands, T ≤ 20 frames) and increase if the model overfits.

---

### 2. Voiced/unvoiced imbalance: beyond uniform BCE

In most singing datasets, **unvoiced (silent) frames outnumber voiced ones by 2–5×**. With uniform BCE, the model can achieve low loss just by predicting "unvoiced" most of the time, ignoring pitch entirely.

**What's already in the code.** `train.py` weights the pitch loss by VAD (silent frames contribute zero pitch gradient). This helps but does not fix VAD itself — the VAD head still sees imbalanced labels.

**Things to try:**

1. **Positive weighting.** Pass `pos_weight` to `nn.BCEWithLogitsLoss` to upweight voiced frames:

   ```python
   # If ~30% of frames are voiced, pos_weight ≈ 0.7 / 0.3 ≈ 2.3
   vad_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.3]).to(device))
   ```

2. **Focal loss \[8\].** Focal loss multiplies the per-example BCE by `(1 − p_t)^γ`, automatically down-weighting easy negatives (clear silence) and concentrating the gradient on hard examples (ambiguous onset/offset frames). With γ = 2 and a voiced-positive weight, this often tightens VAD boundaries noticeably:

   ```python
   def focal_bce(pred, target, gamma=2.0, pos_weight=2.0):
       bce = F.binary_cross_entropy(pred, target, reduction='none')
       pt = torch.where(target == 1, pred, 1 - pred)
       weight = torch.where(target == 1,
                            torch.full_like(pt, pos_weight), torch.ones_like(pt))
       return (weight * (1 - pt) ** gamma * bce).mean()
   ```

3. **Balanced sampling.** Instead of uniform random clip sampling in `NanoPitchDataset.__getitem__`, sample with probability proportional to the voiced-frame fraction of each segment. This ensures each batch contains a mix of high-activity and low-activity clips.

---

### 3. Longer training segments (context window)

The default `--seq-len 200` gives the GRU 2 seconds of context per gradient step. Pitch is a slowly varying signal — vibrato cycles are ~5–7 Hz, and a phrase can sustain for 3–5 seconds — so longer sequences can help the model learn smoother transitions.

**Try:**

```bash
python train.py --seq-len 400   # 4 seconds
python train.py --seq-len 600   # 6 seconds
```

**Trade-offs:**
- Longer sequences cost more memory (`O(T)` for GRU hidden states and activations). If you run out of RAM, reduce `--batch-size` proportionally (e.g. halve seq-len doubles T, so halve batch-size to keep total tokens constant).
- Very long sequences can make the initial GRU hidden state matter more. You can address this by warming up with a few frames of silence before the actual clip, or by training on randomly truncated segments so the model learns to handle cold starts.
- Watch for gradient vanishing in long GRU sequences (see §5 below).

---

### 4. Train for longer (more gradient steps)

50 epochs at the default settings is a reasonable starting point, but models often continue improving well beyond that. Two strategies:

**More epochs:**

```bash
python train.py --epochs 200
```

**Resume from a checkpoint:**

```bash
python train.py --resume ./runs/exp1/checkpoints/best.pth \
                --epochs 100 --lr 1e-4
```

Fine-tuning from a converged checkpoint with a lower learning rate often recovers another 1–3% RPA, especially on noisy conditions.

**When to stop.** Watch the evaluation RPA at `clean` SNR on TensorBoard. If it has not improved for 20+ epochs and the loss curve is flat, training has converged. If clean RPA improves but noisy RPA stalls, augmentation (§1, §2) is the bottleneck, not training length.

---

### 5. Monitor gradient norm; try cosine annealing

**What is a gradient?** During backpropagation, the loss function is differentiated with respect to every learnable parameter in the model. The result for each parameter is a *gradient* — a scalar that says "if I increase this weight slightly, does the loss go up or down, and by how much?" The optimizer uses these gradients to nudge all parameters in the direction that reduces loss.

**What is the gradient norm?** Collecting all the gradients from every parameter into a single long vector and taking its Euclidean length gives the *gradient norm* — a single number summarising the overall magnitude of the update signal across the whole network. A very large norm means some parameters are receiving a huge push (potentially destabilising training); a very small norm means the model is barely learning.

**Why it matters for GRUs.** GRUs process sequences frame by frame, and gradients must flow backward through every time step. This makes them susceptible to two failure modes \[9\]:
- **Exploding gradients** — the norm grows exponentially with sequence length, causing parameter updates so large they overshoot the optimum and destabilise training.
- **Vanishing gradients** — the norm shrinks toward zero, so early frames in a sequence receive almost no learning signal and long-range dependencies never form.

The code already clips the gradient norm (`clip_grad_norm_`) to prevent explosions, but you should *log* the pre-clip norm to TensorBoard so you can see what is actually happening during training:

```python
# After loss.backward(), before optimizer.step():
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
writer.add_scalar("train/grad_norm", total_norm, global_step)
```

Interpret the resulting TensorBoard curve:

- **Norm consistently at the clip ceiling** — the clip is active every step; the learning rate is probably too high.
- **Norm collapses toward zero after a few epochs** — vanishing gradient; try shorter sequences (§3) or a smaller model.
- **Norm spikes on occasional batches but is otherwise stable** — a few hard batches hitting the clip. This is normal; clipping handles it.

**Cosine annealing.** The default scheduler holds the learning rate constant (your stub from §4 is a starting point). Cosine annealing with warm restarts \[10\] can escape local minima by periodically resetting the learning rate to a higher value, then smoothly decaying it again:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-5)
```

Each restart gives the optimizer a fresh running start that often finds a better basin. Log `scheduler.get_last_lr()[0]` to TensorBoard to visualise the schedule alongside the gradient norm.

---

### 6. Train heads separately; freeze one while training the other

NanoPitch has two output heads sharing a common backbone (the GRU stack):

- **VAD head**: easy binary classification — clean task, fast to converge.
- **Pitch head**: 360-class posterior — much harder, needs more gradient signal.

This asymmetry means both heads fight over the shared gradient during joint training. A curriculum strategy can help:

**Phase 1 — pre-train VAD only.** Freeze the pitch head and train the backbone + VAD head for a few epochs. This establishes a stable voiced/unvoiced representation before the pitch head introduces its noisy gradient:

```python
# Freeze pitch head
for p in model.dense_pitch.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
# Train for ~10 epochs, then unfreeze
```

**Phase 2 — freeze VAD, train pitch.** Once VAD has converged, freeze it and concentrate all gradient on the pitch head:

```python
for p in model.dense_vad.parameters():
    p.requires_grad = False
for p in model.dense_pitch.parameters():
    p.requires_grad = True
```

**Phase 3 — joint fine-tuning.** Unfreeze everything and fine-tune jointly with a lower learning rate.

This staged approach is a form of **curriculum learning** and can significantly improve pitch accuracy, particularly in noisy conditions where early VAD errors corrupt the pitch training signal. You can monitor per-head loss in TensorBoard (`train/vad` and `train/pitch`) to decide when to transition between phases.

---

## Project Structure

```
nanopitch/
  data/
    clean.npz          Pre-extracted clean vocal features
    noise.npz          Pre-extracted noise features
    test.npz           Test clips at multiple noise levels
  training/
    model.py           Neural network definition (start reading here!)
    train.py           Training loop (augmentation: implement augment_mel_batch)
    evaluate.py        Standalone evaluation with detailed report
  deployment/
    export_weights.py  Convert PyTorch checkpoint -> JSON for browser
    wasm/
      nanopitch.h      C inference engine header
      nanopitch.c      C inference engine (mel + GRU + Viterbi)
      build.sh         Compile C to WebAssembly
    web/
      index.html       Live pitch tracking web app
```

## Suggested Reading Order

1. **`training/model.py`** — understand the neural network architecture, pitch representation, and Viterbi decoding
2. **`training/train.py`** — training loop; implement `augment_mel_batch` for noise mixing
3. **`training/evaluate.py`** — understand the evaluation metrics and offline vs realtime comparison
4. **`deployment/export_weights.py`** — how PyTorch weights become a JSON file for the browser
5. **`deployment/wasm/nanopitch.c`** — how the same model runs in C (mel FFT, GRU cells, Viterbi)
6. **`deployment/web/index.html`** — how the browser captures audio and runs WASM inference

## Data Sources

- **GTSinger** \[1\] — Studio-recorded singing with MIDI annotations (9 languages), pitch ground truth extracted with RMVPE \[2\]
- **FSDNoisy18k** \[3\] — Environmental noise for augmentation and testing

## References

\[1\] Zhang, Y., Pan, C., Guo, W., Li, R., Zhu, Z., Wang, J., Xu, W., Lu, J., Hong, Z., Wang, C., Zhang, L., He, J., Jiang, Z., Chen, Y., Yang, C., Zhou, J., Cheng, X., & Zhao, Z. (2024). GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks. *Advances in Neural Information Processing Systems (NeurIPS 2024)*. arXiv:2409.13832.

\[2\] Wei, H., Cao, X., Dan, T., & Chen, Y. (2023). RMVPE: A Robust Model for Vocal Pitch Estimation in Polyphonic Music. *Proc. Interspeech 2023*, 5421–5425. doi:10.21437/Interspeech.2023-528.

\[3\] Fonseca, E., Plakal, M., Ellis, D. P. W., Font, F., Favory, X., & Serra, X. (2019). Learning Sound Event Classifiers from Web Audio with Noisy Labels. *Proc. IEEE ICASSP 2019*. arXiv:1901.01189.

\[4\] Valin, J.-M. (2018). A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band Speech Enhancement. *Proc. IEEE MMSP 2018*. arXiv:1709.08243. *(Architecture inspiration for NanoPitch)*

\[5\] Cho, K., van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. *Proc. EMNLP 2014*, 1724–1734. arXiv:1406.1078. *(Original GRU paper)*

\[6\] Viterbi, A. (1967). Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm. *IEEE Transactions on Information Theory*, 13, 260–269. doi:10.1109/TIT.1967.1054010.

\[7\] Park, D. S., Chan, W., Zhang, Y., Chiu, C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. *Proc. Interspeech 2019*, 2613–2617. doi:10.21437/Interspeech.2019-2680. arXiv:1904.08779.

\[8\] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *Proc. ICCV 2017*, 2980–2988. arXiv:1708.02002.

\[9\] Pascanu, R., Mikolov, T., & Bengio, Y. (2013). On the Difficulty of Training Recurrent Neural Networks. *Proc. ICML 2013*, Vol. 28, 1310–1318.

\[10\] Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. *Proc. ICLR 2017*. arXiv:1608.03983.
