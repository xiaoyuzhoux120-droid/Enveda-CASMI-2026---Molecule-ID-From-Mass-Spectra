# Ours · 4: analog-evidence re-rank of ranks 2-25 (modified entropy, research/spectral_matching).
# Runs last. Rank 1 never moves; the candidate set never changes. On any error, guard or failed check the
# input submission.csv is left byte for byte as it was.
import os as _al_os, sys as _al_sys, json as _al_json, time as _al_time, traceback as _al_tb, hashlib as _al_hashlib
from pathlib import Path as _AlPath
import pandas as _al_pd, numpy as _al_np
_AL_LAM = 0.35
_AL_START_LIMIT_S = 8.45 * 3600     # do not start after this much session time
_AL_HARD_LIMIT_S = 8.80 * 3600      # abort (keep input) if still running at this session time
_AL_SRC = __AL_SOURCE__
_al_audit = dict(lam=_AL_LAM, status='not run', changed_molecules=0, molecules=0)
_al_sub = _AlPath(ROOT) / 'submission.csv'
_al_before = _al_sub.read_bytes()
try:
    _al_el = _al_time.time() - NB_T0
    if _al_el > _AL_START_LIMIT_S:
        raise TimeoutError(f'session time {_al_el:.0f}s past start limit')
    _al_mod = _AlPath(ROOT) / 'analog_layer.py'
    _al_mod.write_text(_AL_SRC)
    _al_sys.path.insert(0, str(ROOT))
    import importlib as _al_il
    _al = _al_il.import_module('analog_layer')
    _al_t0 = _al_time.time()
    _al_lib = _al.build_library(str(COMP / 'train.parquet'))
    _al_te = _al_pd.read_parquet(COMP / 'test.parquet')
    _al_df = _al_pd.read_csv(_al_sub, dtype=str, keep_default_na=False)
    _al_groups = {str(k): g for k, g in _al_te.groupby('molecule_id', sort=False)}
    _al_new = []
    for _al_mid, _al_text in zip(_al_df.molecule_id, _al_df.smiles):
        if _al_time.time() - NB_T0 > _AL_HARD_LIMIT_S:
            raise TimeoutError('hard session limit reached')
        _al_c = _al_text.split(';')
        _al_g = _al_groups.get(str(_al_mid))
        if _al_g is None or len(_al_c) < 3:
            _al_new.append(_al_text); continue
        _al_an = _al.analogs(_al_lib, _al_g)
        _al_ev = _al.evidence(_al_lib, _al_an, _al_c)
        _al_ord = _al.rerank(_al_c, _al_ev, _AL_LAM)
        _al_out = [_al_c[i] for i in _al_ord]
        assert _al_out[0] == _al_c[0] and sorted(_al_out) == sorted(_al_c) and len(_al_out) == len(_al_c)
        _al_audit['changed_molecules'] += int(_al_out != _al_c)
        _al_new.append(';'.join(_al_out))
    _al_res = _al_df.copy(); _al_res['smiles'] = _al_new
    assert list(_al_res.molecule_id) == list(_al_df.molecule_id)
    assert _al_res.smiles.str.split(';').map(len).tolist() == _al_df.smiles.str.split(';').map(len).tolist()
    assert (_al_res.smiles.str.split(';').str[0] == _al_df.smiles.str.split(';').str[0]).all()
    if _al_sub.read_bytes() != _al_before:
        raise RuntimeError('submission.csv changed underneath the layer')
    _al_tmp = _AlPath(ROOT) / 'submission.analog.tmp.csv'
    _al_res.to_csv(_al_tmp, index=False)
    _al_os.replace(_al_tmp, _al_sub)
    _al_audit.update(status='applied', molecules=len(_al_res), seconds=round(_al_time.time() - _al_t0, 1),
                     library_reps=int(len(_al_lib['prec'])))
except Exception as _al_e:
    if _al_sub.read_bytes() != _al_before:
        _al_sub.write_bytes(_al_before)
    _al_audit.update(status='kept input', exception=repr(_al_e), trace=_al_tb.format_exc()[-2000:])
(_AlPath(ROOT) / 'analog_layer_audit.json').write_text(_al_json.dumps(_al_audit, indent=2, default=str))
print('ANALOG LAYER', _al_json.dumps({k: _al_audit.get(k) for k in ('status', 'lam', 'molecules', 'changed_molecules', 'seconds', 'exception')}), flush=True)
