# NanoPitch experiment tracker

Tracked metrics (realtime Viterbi, matches browser deployment):
- **VAD Acc** — VAD head accuracy, `(pv > 0.5) == (vr > 0.5)`. Decoder-independent.
- **rtRPA / rtVDR / rtMed¢** — realtime greedy Viterbi.

Checkpoint reported: `best_macro_rpa.pth` unless noted. To auto-populate a row:

```bash
cd training
python update_results.py --run-dir ./runs/<run_name> --data-dir ../data \
    --name <row_label> --note "what you expected"

# Remove a row (also removes the matching per-condition entry):
python update_results.py --delete <row_label>
```

## Runs

| # | Run | Key args / what changed | Expectation | VAD Acc | rtRPA | rtVDR | rtMed¢ |
|---|---|---|---|---|---|---|---|
| 1 | `baseline` | defaults | _tbd_ | 79.4 | 91.1 | 65.9 | 29.6 |
| 2 | `wpitch2.0` | `--w-pitch`=2 | RPA should increase as model now prioritizes pitch over VAD 2 times more than baseline | 80.2 (+0.8) | 91.6 (+0.5) | 69.3 (+3.4) | 31.0 (+1.4) |
| 3 | `gru128` | `--gru-size`=128 | _tbd_ | 79.3 (-0.1) | 88.9 (-2.2) | 70.9 (+5.0) | 37.9 (+8.3) |
| 4 | `noise` | `--augment`=noise | _tbd_ | 81.7 (+2.3) | 91.3 (+0.2) | 66.5 (+0.6) | 36.1 (+6.5) |
| 5 | `noise_wpitch2.0_cosine` | `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise | _tbd_ | 81.6 (+2.2) | 93.9 (+2.8) | 67.8 (+1.9) | 23.0 (-6.6) |
| 6 | `specnoiseaug_wpitch2.0_cosine` | `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug | _tbd_ | 81.3 (+1.9) | 93.0 (+1.9) | 55.6 (-10.3) | 36.5 (+6.9) |
| 7 | `specnoiseaug_wpitch2_wvad2_cosine` | `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug | _tbd_ | 81.4 (+2.0) | 94.1 (+3.0) | 46.8 (-19.1) | 27.2 (-2.4) |
| 8 | `specnoiseaug_wpitch2.0_cosine_snrn10to30` | `--epochs`=100, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.5 (+2.1) | 93.9 (+2.8) | 67.6 (+1.7) | 22.2 (-7.4) |
| 9 | `specaug_wpitch2.0_cosine_100Epochs_n10t30snr_seq400_batch128` | `--seq-len`=400, `--epochs`=100, `--batch-size`=128, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.5 (+2.1) | 93.7 (+2.6) | 49.6 (-16.3) | 34.7 (+5.1) |
| 10 | `specaug_wpitch2.0_cosine_100Epochs_n10t30snr_seq400` | `--seq-len`=400, `--epochs`=100, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.8 (+2.4) | 94.7 (+3.6) | 62.1 (-3.8) | 18.9 (-10.7) |
| 11 | `specaug_wpitch2.0_cosine_100Epochs_snr07_n10t30snr_seq400` | `--seq-len`=400, `--epochs`=100, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-bias`=0.7, `--snr-range`=-10 30 | _tbd_ | 81.7 (+2.3) | 94.3 (+3.2) | 60.6 (-5.3) | 27.1 (-2.5) |
| 12 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.7 (+2.3) | 95.0 (+3.9) | 65.1 (-0.8) | 13.2 (-16.4) |
| 13 | `seq800_btch12_wVad0.2wPitch2_cosine_specaug_n10t30SNR` | `--seq-len`=800, `--epochs`=100, `--batch-size`=12, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.7 (+2.3) | 93.8 (+2.7) | 68.5 (+2.6) | 21.8 (-7.8) |
| 14 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_strict` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-mask`=strict | _tbd_ | 81.9 (+2.5) | 94.8 (+3.7) | 66.7 (+0.8) | 17.9 (-11.7) |
| 15 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_poswvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--vad-loss`=pos_weight, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 80.2 (+0.8) | 93.3 (+2.2) | 63.4 (-2.5) | 28.4 (-1.2) |
| 16 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_poswvad0.6` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--vad-loss`=pos_weight, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 82.4 (+3.0) | 94.1 (+3.0) | 68.9 (+3.0) | 16.7 (-12.9) |
| 17 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_vadfocal` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 82.2 (+2.8) | 94.5 (+3.4) | 75.6 (+9.7) | 15.0 (-14.6) |
| 18 | `seq600_btch16_wVad0.2wPitch2_cosinerestart_specaug_n10t30SNR` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmrestart, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.6 (+2.2) | 94.6 (+3.5) | 62.2 (-3.7) | 27.2 (-2.4) |
| 19 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_freqmask2` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--freq-mask-param`=2 | _tbd_ | 81.6 (+2.2) | 93.3 (+2.2) | 66.8 (+0.9) | 22.0 (-7.6) |
| 20 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_curr` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--curriculum`=on, `--curriculum-vad-epochs`=20, `--curriculum-pitch-epochs`=40 | _tbd_ | 81.0 (+1.6) | 93.0 (+1.9) | 55.1 (-10.8) | 39.0 (+9.4) |
| 21 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_restart100` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--resume`=seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR | _tbd_ | 81.7 (+2.3) | 94.9 (+3.8) | 65.1 (-0.8) | 12.6 (-17.0) |
| 22 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_balanced` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--balanced-sampling`=on, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 81.8 (+2.4) | 93.8 (+2.7) | 68.0 (+2.1) | 27.6 (-2.0) |
| 23 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_focal1` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--vad-loss`=focal, `--vad-focal-gamma`=1, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 80.3 (+0.9) | 94.7 (+3.6) | 70.5 (+4.6) | 13.4 (-16.2) |
| 24 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 81.7 (+2.3) | 97.1 (+6.0) | 40.9 (-25.0) | 6.3 (-23.3) |
| 25 | `seq600_btch16_wVad0.1wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 81.6 (+2.2) | 96.9 (+5.8) | 58.7 (-7.2) | 12.1 (-17.5) |
| 26 | `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.05, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 81.6 (+2.2) | 96.1 (+5.0) | 65.9 (+0.0) | 15.8 (-13.8) |
| 27 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.2, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 96.0 (+16.6) | 96.6 (+5.5) | 38.0 (-27.9) | 13.6 (-16.0) |
| 28 | `seq600_btch16_wVad0.01wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.01, `--w-pitch`=2, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 94.8 (+15.4) | 95.9 (+4.8) | 65.9 (+0.0) | 23.6 (-6.0) |
| 29 | `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_focal_vadposw0.6` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.05, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 95.3 (+15.9) | 95.8 (+4.7) | 66.6 (+0.7) | 25.6 (-4.0) |
| 30 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 90.9 (+11.5) | 95.9 (+4.8) | 69.9 (+4.0) | 24.0 (-5.6) |
| 31 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_focal_posw04` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.4, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 90.1 (+10.7) | 95.7 (+4.6) | 70.0 (+4.1) | 24.8 (-4.8) |
| 32 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30 | _tbd_ | 90.9 (+11.5) | 94.3 (+3.2) | 77.1 (+11.2) | 25.5 (-4.1) |
| 33 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.4_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.4 | _tbd_ | 90.8 (+11.4) | 97.8 (+6.7) | 54.4 (-11.5) | 14.1 (-15.5) |
| 34 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma1.0_cleanvad` | `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=1 | _tbd_ | 90.5 (+11.1) | 95.2 (+4.1) | 74.3 (+8.4) | 22.6 (-7.0) |
| 35 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma1.0_cleanvad_gru128` | `--gru-size`=128, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=1 | _tbd_ | 91.9 (+12.5) | 95.4 (+4.3) | 75.9 (+10.0) | 21.4 (-8.2) |
| 36 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128` | `--gru-size`=128, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8 | _tbd_ | 91.5 (+12.1) | 96.1 (+5.0) | 71.9 (+6.0) | 20.9 (-8.7) |
| 37 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128-resume` | `--gru-size`=128, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--lr`=0.0003, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8, `--resume`=seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128 | _tbd_ | 91.5 (+12.1) | 96.1 (+5.0) | 71.7 (+5.8) | 20.5 (-9.1) |
| 38 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128-resume2` | `--gru-size`=128, `--seq-len`=600, `--epochs`=100, `--batch-size`=16, `--w-vad`=0.001, `--w-pitch`=2, `--vad-loss`=focal, `--vad-pos-weight`=0.6, `--scheduler`=cosine_warmup, `--vad-target`=clean_vad, `--augment`=noise_specaug, `--snr-range`=-10 30, `--pitch-sigma`=0.8, `--resume`=seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128 | _tbd_ | 91.3 (+11.9) | 96.2 (+5.1) | 71.7 (+5.8) | 20.6 (-9.0) |

## Per-condition rtRPA

Realtime (browser) RPA per SNR condition. "Macro" is the mean over all clips (= overall line in `evaluate.py`). Use this table to see where a run wins/loses — low-SNR or clean.

| # | Run | -5 dB | +0 dB | +5 dB | +10 dB | +20 dB | clean | Macro |
|---|---|---|---|---|---|---|---|---|
| 33 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.4_cleanvad` | 96.0 | 96.6 | 96.8 | 98.7 | 99.2 | 99.3 | 97.8 |
| 24 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | 97.2 | 96.0 | 95.8 | 97.8 | 97.9 | 97.8 | 97.1 |
| 25 | `seq600_btch16_wVad0.1wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | 96.7 | 95.4 | 95.4 | 97.7 | 97.9 | 98.5 | 96.9 |
| 27 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | 95.6 | 94.1 | 95.5 | 97.8 | 97.8 | 98.4 | 96.6 |
| 38 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128-resume2` | 94.6 | 93.5 | 96.0 | 97.6 | 97.5 | 97.9 | 96.2 |
| 26 | `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8` | 95.1 | 94.3 | 94.4 | 96.9 | 97.9 | 98.1 | 96.1 |
| 36 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128` | 94.2 | 93.7 | 95.5 | 97.5 | 97.6 | 98.0 | 96.1 |
| 37 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_gru128-resume` | 94.2 | 93.6 | 95.5 | 97.6 | 97.6 | 98.1 | 96.1 |
| 28 | `seq600_btch16_wVad0.01wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | 93.6 | 93.1 | 96.1 | 97.4 | 96.6 | 98.2 | 95.9 |
| 30 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad` | 94.8 | 93.1 | 95.8 | 97.6 | 96.3 | 98.0 | 95.9 |
| 29 | `seq600_btch16_wVad0.05wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_focal_vadposw0.6` | 94.3 | 93.5 | 95.2 | 97.6 | 95.8 | 98.1 | 95.8 |
| 31 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma0.8_cleanvad_focal_posw04` | 94.4 | 93.4 | 95.0 | 97.1 | 96.3 | 97.6 | 95.7 |
| 35 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma1.0_cleanvad_gru128` | 94.0 | 92.9 | 95.2 | 96.9 | 96.0 | 97.4 | 95.4 |
| 34 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_pitchsigma1.0_cleanvad` | 94.3 | 92.0 | 95.2 | 96.7 | 95.3 | 97.4 | 95.2 |
| 12 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR` | 92.8 | 93.2 | 94.8 | 95.7 | 96.5 | 96.8 | 95.0 |
| 21 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_restart100` | 92.7 | 93.3 | 94.6 | 95.4 | 96.5 | 96.7 | 94.9 |
| 14 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_strict` | 92.5 | 93.9 | 94.4 | 96.0 | 95.1 | 97.1 | 94.8 |
| 10 | `specaug_wpitch2.0_cosine_100Epochs_n10t30snr_seq400` | 92.8 | 93.0 | 94.3 | 95.6 | 95.4 | 97.3 | 94.7 |
| 23 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_focal1` | 91.9 | 93.1 | 95.2 | 94.5 | 96.5 | 96.6 | 94.7 |
| 18 | `seq600_btch16_wVad0.2wPitch2_cosinerestart_specaug_n10t30SNR` | 92.7 | 93.5 | 93.2 | 96.1 | 95.2 | 96.9 | 94.6 |
| 17 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_vadfocal` | 93.0 | 93.8 | 93.5 | 95.3 | 94.2 | 96.9 | 94.5 |
| 11 | `specaug_wpitch2.0_cosine_100Epochs_snr07_n10t30snr_seq400` | 93.0 | 92.1 | 93.9 | 95.0 | 94.4 | 97.3 | 94.3 |
| 32 | `seq600_btch16_wVad0.001wPitch2_cosine_specaug_n10t30SNR_cleanvad` | 93.0 | 90.8 | 94.3 | 95.6 | 95.2 | 96.9 | 94.3 |
| 7 | `specnoiseaug_wpitch2_wvad2_cosine` | 92.8 | 92.4 | 92.4 | 95.5 | 94.4 | 97.0 | 94.1 |
| 16 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_poswvad0.6` | 93.0 | 92.0 | 93.1 | 95.6 | 93.8 | 97.0 | 94.1 |
| 5 | `noise_wpitch2.0_cosine` | 92.3 | 90.8 | 92.6 | 95.4 | 95.6 | 96.4 | 93.9 |
| 8 | `specnoiseaug_wpitch2.0_cosine_snrn10to30` | 91.9 | 90.3 | 93.2 | 95.9 | 95.0 | 97.1 | 93.9 |
| 13 | `seq800_btch12_wVad0.2wPitch2_cosine_specaug_n10t30SNR` | 89.6 | 93.9 | 93.7 | 94.5 | 95.1 | 95.8 | 93.8 |
| 22 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_balanced` | 92.7 | 91.2 | 93.0 | 95.4 | 94.8 | 95.9 | 93.8 |
| 9 | `specaug_wpitch2.0_cosine_100Epochs_n10t30snr_seq400_batch128` | 90.9 | 91.5 | 93.1 | 95.9 | 94.4 | 96.1 | 93.7 |
| 15 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_poswvad` | 90.2 | 90.6 | 94.5 | 94.0 | 95.1 | 95.3 | 93.3 |
| 19 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_freqmask2` | 92.3 | 91.9 | 93.1 | 94.0 | 94.9 | 93.7 | 93.3 |
| 6 | `specnoiseaug_wpitch2.0_cosine` | 90.5 | 89.5 | 90.7 | 95.2 | 94.8 | 97.0 | 93.0 |
| 20 | `seq600_btch16_wVad0.2wPitch2_cosine_specaug_n10t30SNR_curr` | 90.6 | 90.1 | 92.9 | 95.1 | 93.8 | 94.8 | 93.0 |
| 2 | `wpitch2.0` | 89.0 | 89.5 | 88.5 | 92.3 | 93.9 | 96.0 | 91.6 |
| 4 | `noise` | 88.7 | 87.9 | 89.8 | 93.6 | 92.8 | 94.5 | 91.3 |
| 1 | `baseline` | 88.1 | 90.6 | 90.2 | 89.3 | 91.9 | 95.9 | 91.1 |
| 3 | `gru128` | 84.6 | 89.4 | 88.4 | 88.3 | 89.4 | 93.3 | 88.9 |

## Change log

_(fill in as runs land)_

## Planned experiments (from README §1–§6)

| # | Idea | Knob / code area | Expectation |
|---|---|---|---|
| P1 | Log-domain noise mixing + SpecAugment | `--augment noise_specaug`, tune `--snr-range`, `--p-clean`, `--snr-bias` | robustness across SNR levels → lift macro RPA without hurting clean |
| P2 | Voiced/unvoiced imbalance: `pos_weight` or focal BCE on VAD head | modify VAD BCE in `train.py` (or switch to `BCEWithLogitsLoss(pos_weight=…)`) | sharper VAD boundaries → higher VAD Acc + VDR |
| P3 | Longer context window | `--seq-len 400` / `600` (halve `--batch-size`) | smoother pitch transitions; better low-SNR RPA |
| P4 | Train longer / resume fine-tune | `--epochs 150–200`, or `--resume … --lr 1e-4 --epochs 50` | +1–3% RPA, most noticeable on noisy SNR |
| P5 | Scheduler + grad-norm monitoring | `--scheduler cosine_warmup` / `cosine_warmrestart` (with `--lr-t0`) | escape local minima; confirm clip ceiling not saturating |
| P6 | Curriculum (VAD → pitch → joint) | `--curriculum --curriculum-vad-epochs N --curriculum-pitch-epochs M` | clean VAD representation before pitch gradient enters; better noisy-condition pitch |
