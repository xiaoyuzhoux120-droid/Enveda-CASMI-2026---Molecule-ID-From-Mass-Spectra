"""Recover completed rounds from this experiment's budget-only failure.

Does not resume optimizer/RNG, consume partial epoch weights, or reuse the
older scientific experiment. Recomputes predictions from selected binary.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from casmi_ml.np_pairtail_fpnet import FPNet,predict,sha256
from casmi_ml.data import write_json


def recover_completed_seed(source,root,train,dev,bits,weights,protocol,declaration):
    source,root=Path(source),Path(root)
    expected=declaration['files_sha256']
    for name,digest in expected.items():
        if sha256(source/name)!=digest:raise ValueError('Recovered seed file SHA differs: '+name)
    state=json.loads((source/'run_status.json').read_text())
    if (state['error_type']!='TimeoutError' or state['error']!='Seed 20261004 exceeded frozen 45-minute budget'
            or state['acceptance_opened'] or state['competition_submission_allowed']):
        raise ValueError('Only declared budget-only failure can be recovered')
    seed=20261004;history=json.loads((source/f'seed_{seed}_history.json').read_text())
    if not protocol['fpnet']['minimum_epochs']<=len(history)<=protocol['fpnet']['maximum_epochs']:
        raise ValueError('Recovered seed minimum/maximum completed epoch violation')
    if [r['epoch'] for r in history]!=list(range(1,len(history)+1)):
        raise ValueError('Missing completed epoch in recovery history')
    checkpoint=torch.load(source/f'fpnet_seed_{seed}.pt',map_location='cpu',weights_only=True)
    best=min(history,key=lambda row:row['development']['bce'])
    if (checkpoint['seed']!=seed or checkpoint['selected_epoch']!=best['epoch']
            or checkpoint['optimizer_updates']!=best['optimizer_updates']
            or checkpoint['development_bce']!=best['development']['bce']
            or checkpoint['bits']!=bits.tolist() or checkpoint['preprocessing']!=train.layout['preprocessing']
            or checkpoint['preprocessing']!=dev.layout['preprocessing']
            or checkpoint['metadata_dim']!=train.arrays['meta'].shape[1]
            or checkpoint['architecture']!={'layers':6,'hidden':512,'heads':8,'dropout':.1}):
        raise ValueError('Recovered selected model disagrees with frozen completed history')
    raw=FPNet(checkpoint['metadata_dim'],len(bits)).to('cuda:0');raw.load_state_dict(checkpoint['state_dict'])
    if any(not torch.isfinite(p).all() for p in raw.parameters()):raise ValueError('Recovered nonfinite weight')
    model=nn.DataParallel(raw) if torch.cuda.device_count()>1 else raw
    single,fused=predict(model,dev,bits,torch.device('cuda:0'))
    differences={}
    for name,value in [('single',single),('fused',fused)]:
        recorded=np.load(source/f'seed_{seed}_{name}_logits.npy')
        if value.shape!=recorded.shape or not np.isfinite(recorded).all():raise ValueError('Recovered development logits invalid')
        difference=float(np.max(np.abs(value-recorded)));differences[name]=difference
        if not np.allclose(value,recorded,rtol=1e-5,atol=1e-4):
            raise ValueError('Recovered binary prediction replay differs from saved logits')
        # Keep original saved arrays after verifying exact selected-binary replay.
        shutil.copy2(source/f'seed_{seed}_{name}_logits.npy',root/f'seed_{seed}_{name}_logits.npy')
    loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,dtype=torch.float32))
    observed=float(loss(torch.from_numpy(fused),torch.from_numpy(np.asarray(dev.arrays['target'][:,bits],np.float32))))
    if abs(observed-best['development']['bce'])>1e-6:raise ValueError('Recovered selected weighted BCE mismatch')
    for name in [f'fpnet_seed_{seed}.pt',f'seed_{seed}_history.json']:
        shutil.copy2(source/name,root/name)
    audit={'stage':'M2_completed_seed_recovered','seed':seed,'completed_epochs':len(history),
        'selected_epoch':checkpoint['selected_epoch'],'selected_weight_updates':checkpoint['optimizer_updates'],
        'completed_epoch_updates':history[-1]['optimizer_updates'],'partial_epoch_20_used':False,
        'optimizer_resume':False,'prediction_replay_max_difference':differences,
        'recomputed_bce':observed,'files_sha256':expected,'failed_worker_seconds_already_charged':2840.,
        'source_script_version':355536163,'acceptance_opened':False}
    write_json(root/'seed_recovery_audit.json',audit);print(json.dumps(audit),flush=True)
    del model,raw,checkpoint;torch.cuda.empty_cache()
    return {'seed':seed,'completed_epochs':len(history),'selected_bce':best['development']['bce'],
        'actual_optimizer_updates':history[-1]['optimizer_updates'],'seconds':history[-1]['elapsed_seconds'],
        'stop_reason':'recovered_completed_epochs_from_budget_failure',
        'checkpoint_sha256':sha256(root/f'fpnet_seed_{seed}.pt'),'recovery_audit_sha256':sha256(root/'seed_recovery_audit.json')}
