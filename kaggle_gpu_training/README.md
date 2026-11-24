# Kaggle GPU training kernel (template, not pushed)

Private training script that runs `scripts/train_fp_gpu.py` on a Kaggle GPU with
competition data attached. This is a **training** kernel, separate from the
internet-off submission notebooks; it never writes `submission.csv` and never submits.

Only the owner pushes it, after review:

```bash
cp scripts/train_fp_gpu.py kaggle_gpu_training/
.venv/bin/kaggle kernels push -p kaggle_gpu_training      # creates/updates a PRIVATE kernel
.venv/bin/kaggle kernels status giaok246/casmi-2026-fp-gpu-training
.venv/bin/kaggle kernels output giaok246/casmi-2026-fp-gpu-training -p artifacts/fp_gpu_kaggle
```

Set the accelerator (e.g. T4 x2 / P100) in the Kaggle editor if the default is not wanted.
Check the weekly GPU quota first (`get_accelerator_quota` in the Kaggle MCP, or the Kaggle UI).
