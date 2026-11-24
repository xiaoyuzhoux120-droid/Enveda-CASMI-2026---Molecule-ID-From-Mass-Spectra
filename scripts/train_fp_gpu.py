"""Spectrum -> Morgan fingerprint trainer for a local GPU or a Kaggle GPU notebook.

Self-contained: reads train.parquet, builds binned-peak + metadata features,
labels each spectrum with a 2048-bit Morgan fingerprint of its structure, uses the
project's molecule-disjoint split (SHA256("42:" + inchikey14) bucket; 0-7999 train,
8000-8999 dev), and trains a residual MLP with mixed precision when CUDA exists.

Examples
  # local GPU, 60K training molecules
  .venv/bin/python scripts/train_fp_gpu.py --data data/train.parquet --max-molecules 60000 --out artifacts/fp_gpu_run1
  # CPU smoke test (what the cloud session runs)
  .venv/bin/python scripts/train_fp_gpu.py --data data/train.parquet --max-molecules 500 --epochs 2 --out /tmp/fp_smoke
  # Kaggle GPU notebook: --data /kaggle/input/enveda-CASMI26-molecule-id-mass-spectra/train.parquet --out /kaggle/working/fp

It only writes into --out. It never contacts Kaggle or submits anything.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

KAGGLE_INPUT = Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra")
if KAGGLE_INPUT.exists():  # Kaggle training kernel (internet on): install the metric-matched RDKit
    try:
        import rdkit  # noqa: F401
        assert rdkit.__version__ == "2026.03.3"
    except (ImportError, AssertionError):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit==2026.3.3"], check=True)
    if len(sys.argv) == 1:  # defaults when Kaggle runs the script without arguments
        sys.argv += ["--data", str(KAGGLE_INPUT / "train.parquet"), "--out", "/kaggle/working/fp_gpu",
                     "--max-molecules", os.environ.get("CASMI_MAX_MOLECULES", "60000")]

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")
COLUMNS = ["inchikey14", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities",
           "precursor_mz", "ionization_mode", "collision_energy_ev"]
N_BINS, MAX_MZ, FP_BITS = 1000, 1000.0, 2048


def bucket(key):
    return int(hashlib.sha256(f"42:{key}".encode()).hexdigest()[:8], 16) % 10000


def load(path, max_molecules, max_spectra_per_mol, seed):
    table = pq.read_table(path, columns=COLUMNS).to_pandas()
    table["bucket"] = table["inchikey14"].map(bucket)
    table = table[table["bucket"] < 9000]  # never touch the holdout range
    rng = np.random.default_rng(seed)
    keys = table["inchikey14"].unique()
    keys = rng.permutation(keys)
    train_keys = [k for k in keys if bucket(k) < 8000][:max_molecules]
    dev_keys = [k for k in keys if 8000 <= bucket(k) < 9000][:max(200, max_molecules // 8)]
    keep = set(train_keys) | set(dev_keys)
    table = table[table["inchikey14"].isin(keep)]
    table = table.groupby("inchikey14", group_keys=False).apply(
        lambda g: g.sample(min(len(g), max_spectra_per_mol), random_state=seed))
    return table.reset_index(drop=True)


def featurize(df):
    x = np.zeros((len(df), N_BINS + 3), np.float32)
    for i, (mzs, its) in enumerate(zip(df["ms2_mzs"], df["ms2_normalized_intensities"])):
        mzs, its = np.asarray(mzs, float), np.asarray(its, float)
        ok = (mzs > 0) & (mzs < MAX_MZ)
        idx = (mzs[ok] / MAX_MZ * N_BINS).astype(int)
        np.maximum.at(x[i, :N_BINS], idx, np.sqrt(its[ok]))
    x[:, N_BINS] = df["precursor_mz"].fillna(0).to_numpy() / MAX_MZ
    x[:, N_BINS + 1] = (df["ionization_mode"] == "positive").astype(np.float32)
    ce = df["collision_energy_ev"].map(lambda v: float(np.mean(v)) if v is not None and len(v) else 0.0)
    x[:, N_BINS + 2] = ce.to_numpy() / 100.0
    return x


def fingerprints(df):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    cache, y = {}, np.zeros((len(df), FP_BITS), np.float32)
    for i, smi in enumerate(df["normalized_smiles"]):
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            cache[smi] = np.array(gen.GetFingerprintAsNumPy(mol), np.float32) if mol else None
        if cache[smi] is not None:
            y[i] = cache[smi]
    return y


class ResidualMLP(torch.nn.Module):
    def __init__(self, d_in, width=1024, depth=3, d_out=FP_BITS, dropout=0.2):
        super().__init__()
        self.inp = torch.nn.Linear(d_in, width)
        self.blocks = torch.nn.ModuleList(torch.nn.Sequential(
            torch.nn.LayerNorm(width), torch.nn.Linear(width, width * 2), torch.nn.GELU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(width * 2, width)) for _ in range(depth))
        self.out = torch.nn.Sequential(torch.nn.LayerNorm(width), torch.nn.Linear(width, d_out))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def tanimoto(logits, y):
    p = (logits > 0).float()
    inter = (p * y).sum(1)
    union = p.sum(1) + y.sum(1) - inter
    return (inter / union.clamp(min=1)).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-molecules", type=int, default=20000)
    ap.add_argument("--spectra-per-molecule", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    df = load(args.data, args.max_molecules, args.spectra_per_molecule, args.seed)
    x, y = featurize(df), fingerprints(df)
    is_dev = (df["bucket"] >= 8000).to_numpy()
    print(f"device={device} spectra={len(df)} train={(~is_dev).sum()} dev={is_dev.sum()} "
          f"prep={time.time() - t0:.1f}s", flush=True)
    xt, yt = torch.from_numpy(x[~is_dev]), torch.from_numpy(y[~is_dev])
    xd, yd = torch.from_numpy(x[is_dev]).to(device), torch.from_numpy(y[is_dev]).to(device)
    model = ResidualMLP(x.shape[1], args.width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=args.epochs * (len(xt) // args.batch + 1))
    use_amp = device == "cuda"
    loss_fn = torch.nn.BCEWithLogitsLoss()
    history, best = [], -1.0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), args.batch):
            idx = perm[i:i + args.batch]
            xb, yb = xt[idx].to(device, non_blocking=True), yt[idx].to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                loss = loss_fn(model(xb).float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            logits = model(xd).float()
            dev_loss, dev_tan = loss_fn(logits, yd).item(), tanimoto(logits, yd)
        history.append({"epoch": epoch + 1, "train_loss": loss.item(), "dev_loss": dev_loss,
                        "dev_tanimoto": dev_tan})
        print(json.dumps(history[-1]), flush=True)
        if dev_tan > best:
            best = dev_tan
            torch.save({"model": model.state_dict(), "args": vars(args), "d_in": x.shape[1]},
                       out / "best.pt")
    (out / "history.json").write_text(json.dumps({"device": device, "args": vars(args),
                                                 "history": history, "seconds": time.time() - t0}, indent=2))
    print(f"best dev tanimoto={best:.4f}; outputs in {out}")


if __name__ == "__main__":
    main()
