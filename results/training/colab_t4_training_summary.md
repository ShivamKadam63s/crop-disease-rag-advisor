# Training results (Colab T4, 15 GB) - transcribed from the notebook output

Setup: Qwen2.5-0.5B-Instruct, 3 epochs, `--bs 1 --accum 16` (default bs 2 ran out of memory in fp32 LoRA at step 7/78 on the T4).
Dataset on Colab built with the real tokenizer (retrieval-miss 12.8%); KB version ad3be9cfa693 (same as the Mac build).

| Mode | trainable params | train_seconds | train_loss | val_loss | peak_gpu_gb |
|---|---|---|---|---|---|
| LoRA (fp32 base) | 8,798,208 / 502,830,976 = 1.75% | 527.7 | 0.1222 | 0.1359 | 8.977 |
| QLoRA (NF4 base) | 8,798,208 / 323,917,696 = 2.72% | 856.4 | 0.1304 | 0.1341 | 3.296 |

Notes for the report:
- QLoRA's trainable percentage is higher only because 4-bit packing shrinks the counted total; the adapter is the same size.
- QLoRA peak memory is about 63% lower than LoRA (3.30 vs 8.98 GB) but training is slower (856 s vs 528 s) because of dequantisation.
- Calibration on Colab: threshold 0.731, in-scope accept 0.96, OOD reject 1.0.
- An earlier attempt with the default batch size gave QLoRA val_loss 0.150 / 0.157 (two runs) and LoRA OOM; those are superseded.

## Second session (same settings, reproducibility check)
| Mode | train_seconds | train_loss | val_loss | peak_gpu_gb |
|---|---|---|---|---|
| LoRA | 553.4 | 0.1216 | 0.1330 | 8.977 |
| QLoRA | 903.3 | 0.1295 | 0.1333 | 3.296 |
Peak memory is identical across sessions; losses differ in the 3rd decimal (GPU non-determinism). Use either session in the report and say which.
