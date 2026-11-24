"""End-to-end environment check for the CASMI 2026 project.

Usage:  .venv/bin/python scripts/verify_env.py [--json results/env_check.json]

Every check is local or read-only. Nothing is submitted, uploaded or published.
Secrets are never printed: only whether a credential is present.
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

RESULTS = {}


def check(name):
    def wrap(fn):
        def run():
            t = time.time()
            try:
                detail = fn()
                RESULTS[name] = {"ok": True, "detail": detail}
            except Exception as exc:  # report, keep going
                RESULTS[name] = {"ok": False, "detail": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc(limit=3)}
            RESULTS[name]["seconds"] = round(time.time() - t, 2)
            mark = "PASS" if RESULTS[name]["ok"] else "FAIL"
            print(f"[{mark}] {name}: {RESULTS[name]['detail']}")
        return run
    return wrap


@check("hardware")
def hardware():
    total, _, free = shutil.disk_usage(".")
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"python": platform.python_version(), "cpus": os.cpu_count(),
            "ram_gb": round(mem_gb, 1), "disk_free_gb": round(free / 1e9, 1), "cuda_gpu": gpu}


@check("versions")
def versions():
    out = {}
    for mod in ["numpy", "pandas", "pyarrow", "scipy", "sklearn", "joblib", "torch", "rdkit",
                "matchms", "pyteomics", "optuna", "jupyterlab", "pytest", "kaggle", "kagglehub"]:
        from importlib.metadata import version
        dist = {"sklearn": "scikit-learn"}.get(mod, mod)
        if mod != "kaggle":  # importing kaggle triggers an auth attempt; metadata is enough
            importlib.import_module(mod)
        out[mod] = version(dist)
    return out


@check("rdkit_metric_matching")
def rdkit_check():
    """Parse SMILES, build Morgan fingerprints, and reproduce the metric's
    tautomer-canonical InChIKey14 match (glucose example from the Evaluation page)."""
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.MolStandardize import rdMolStandardize

    assert rdBase.rdkitVersion == "2026.03.3", f"metric pins 2026.03.3, got {rdBase.rdkitVersion}"
    enum = rdMolStandardize.TautomerEnumerator()

    def key14(smi):
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, smi
        return Chem.MolToInchiKey(enum.Canonicalize(mol))[:14]

    a = key14("OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O")
    b = key14("OCC1OC(O)C(O)C(O)C1O")
    assert a == b == "WQZGKKKJIJFFOK", (a, b)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in
           ["OC(=O)c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"]]
    sim = DataStructs.TanimotoSimilarity(fps[0], fps[1])
    assert 0 < sim < 1
    return {"inchikey14_glucose": a, "tanimoto_salicylic_vs_aspirin": round(sim, 3),
            "morgan_bits_on": fps[0].GetNumOnBits()}


@check("matchms_similarity")
def matchms_check():
    import numpy as np
    from matchms import Spectrum
    from matchms.filtering import default_filters, normalize_intensities
    from matchms.similarity import CosineGreedy, ModifiedCosine

    def spec(mz, it, prec):
        s = Spectrum(mz=np.array(mz, float), intensities=np.array(it, float),
                     metadata={"precursor_mz": prec, "ionmode": "positive"})
        return normalize_intensities(default_filters(s))

    s1 = spec([81.07, 109.06, 137.06, 163.04, 181.05], [0.2, 0.5, 1.0, 0.3, 0.1], 181.05)
    s2 = spec([81.07, 109.06, 137.06, 163.04], [0.25, 0.45, 1.0, 0.35], 181.05)
    s3 = spec([95.08, 123.07, 151.07, 177.05], [0.2, 0.5, 1.0, 0.3], 195.07)  # +14 Da shift
    cos = CosineGreedy(tolerance=0.01).pair(s1, s2)
    mod = ModifiedCosine(tolerance=0.01).pair(s2, s3)
    assert float(cos["score"]) > 0.9 and int(cos["matches"]) >= 4
    assert float(mod["score"]) > 0.9
    return {"cosine_s1_s2": round(float(cos["score"]), 4), "matches": int(cos["matches"]),
            "modified_cosine_shifted": round(float(mod["score"]), 4)}


@check("pytorch_training")
def torch_check():
    """Train a small spectrum->fingerprint MLP on synthetic data; loss must drop."""
    import torch
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(2048, 512, device=device)
    w = torch.randn(512, 256, device=device)
    y = (x @ w > 0).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 256)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.no_grad():
        losses = [loss_fn(model(x), y).item()]  # before training, ~0.69
    for epoch in range(30):
        for i in range(0, len(x), 256):
            opt.zero_grad()
            loss = loss_fn(model(x[i:i + 256]), y[i:i + 256])
            loss.backward()
            opt.step()
        with torch.no_grad():
            losses.append(loss_fn(model(x), y).item())
    with torch.no_grad():
        acc = ((model(x) > 0).float() == y).float().mean().item()
    assert losses[-1] < 0.6 * losses[0] and acc > 0.85, (losses[0], losses[-1], acc)
    return {"device": device, "threads": torch.get_num_threads(), "epochs": 30,
            "loss_before": round(losses[0], 4), "loss_after": round(losses[-1], 4),
            "train_bit_accuracy": round(acc, 4)}


@check("project_tests")
def project_tests():
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4")
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True,
                       text=True, env=env, timeout=1800)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
    assert p.returncode == 0, tail
    return tail


@check("kaggle_mcp_reachable")
def kaggle_mcp():
    sys.path.insert(0, str(Path(__file__).parent))
    import requests
    from check_kaggle_mcp import PROTOCOL, URL, parse
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json",
                      "Accept": "application/json, text/event-stream"})
    if os.environ.get("KAGGLE_API_TOKEN"):
        s.headers["Authorization"] = "Bearer " + os.environ["KAGGLE_API_TOKEN"]
    r = s.post(URL, timeout=30, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": PROTOCOL, "capabilities": {},
        "clientInfo": {"name": "casmi-env-check", "version": "1"}}})
    assert r.status_code == 200, r.status_code
    s.headers["MCP-Protocol-Version"] = PROTOCOL
    r = s.post(URL, timeout=60, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    n = len(parse(r).get("result", {}).get("tools", []))
    # Read-only call: competition metadata (works without auth).
    r = s.post(URL, timeout=60, json={"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
        "name": "get_competition",
        "arguments": {"request": {"competitionName": "enveda-CASMI26-molecule-id-mass-spectra"}}}})
    meta = json.loads(parse(r)["result"]["content"][0]["text"])
    return {"tools": n, "token_present": bool(os.environ.get("KAGGLE_API_TOKEN")),
            "deadline": meta["deadline"], "metric": meta["evaluation_metric"]}


@check("kaggle_cli_auth")
def kaggle_auth():
    """Authenticated read-only call. Fails (expected) until a token is configured."""
    has = bool(os.environ.get("KAGGLE_API_TOKEN")) or Path("~/.kaggle/access_token").expanduser().exists() \
        or Path("~/.kaggle/kaggle.json").expanduser().exists()
    if not has:
        raise RuntimeError("no Kaggle credential configured (see ENVIRONMENT_SETUP.md, manual steps)")
    kaggle = str(Path(sys.executable).parent / "kaggle")
    p = subprocess.run([kaggle, "competitions", "files", "-c",
                        "enveda-CASMI26-molecule-id-mass-spectra"],
                       capture_output=True, text=True, timeout=120)
    out = (p.stdout + p.stderr).strip()
    assert p.returncode == 0 and "403" not in out, out[-300:]
    return out.splitlines()[:6]


@check("competition_data_present")
def data_present():
    files = {f: Path("data", f).exists() for f in ["train.parquet", "test.parquet"]}
    if not all(files.values()):
        raise RuntimeError(f"not downloaded yet: {files} (needs Kaggle credential + accepted rules)")
    import pyarrow.parquet as pq
    return {f: pq.ParquetFile(Path("data", f)).metadata.num_rows for f in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    for fn in [hardware, versions, rdkit_check, matchms_check, torch_check, kaggle_mcp,
               kaggle_auth, data_present]:
        fn()
    if not args.skip_tests:
        project_tests()
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    failed = [k for k, v in RESULTS.items() if not v["ok"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
