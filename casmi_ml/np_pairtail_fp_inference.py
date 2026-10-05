"""Frozen FPNet prediction worker; no fingerprint targets or truth scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from casmi_ml.np_pairtail_fpnet import FPNet,SEEDS,sha256,single_view,merged_views
from casmi_ml.data import write_json


class InputViews(Dataset):
    def __init__(self,cache,identities):
        cache=Path(cache)
        source=json.loads((cache/'layout.json').read_text())
        ids={key:i for i,key in enumerate(source['identities'])}
        self.arrays={name:np.load(cache/(name+'.npy'),mmap_mode='r') for name in ['peaks','mask','meta']}
        self.preprocessing=source['preprocessing']
        self.identities=list(identities);self.layout=[];self.source_indices=[]
        for key in identities:
            if key not in ids:raise ValueError('Prediction identity absent from verified cache')
            group={}
            for kind in ('single','merged'):
                indexes=source['views'][ids[key]][kind]
                group[kind]=list(range(len(self.source_indices),len(self.source_indices)+len(indexes)))
                self.source_indices.extend(indexes)
            self.layout.append(group)
    def __len__(self):return len(self.source_indices)
    def __getitem__(self,i):
        return {k:torch.from_numpy(np.array(v[self.source_indices[i]],copy=True)) for k,v in self.arrays.items()}|{'index':i}


class RuntimeViews(Dataset):
    def __init__(self,frame,preprocessing):
        self.identities=[];self.layout=[];self.rows=[];self.preprocessing=preprocessing
        for key,group in frame.groupby('molecule_id',sort=True):
            self.identities.append(str(key));views={}
            rows=group.to_dict('records')
            for kind,values in [('single',[single_view(row,preprocessing) for row in rows]),
                                ('merged',merged_views(rows,preprocessing))]:
                views[kind]=list(range(len(self.rows),len(self.rows)+len(values)))
                self.rows.extend(values)
            self.layout.append(views)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        return {k:torch.from_numpy(np.asarray(v)) for k,v in zip(['peaks','mask','meta'],self.rows[i])}|{'index':i}


def frozen_predict(models_root,dataset,selected_view,output):
    if not torch.cuda.is_available():raise RuntimeError('FPNet inference requires declared GPU worker')
    if selected_view not in ('single','fused'):raise ValueError('Unfrozen fingerprint view policy')
    models_root,output=Path(models_root),Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=json.loads((models_root/'M2_training_summary.json').read_text())
    gate=json.loads((models_root/'M2_development_candidate_gate.json').read_text())
    state=json.loads((models_root/'run_status.json').read_text())
    if state['stage']!='M2_complete' or not summary['bce_gate_pass'] or not gate['passed']:
        raise ValueError('Incomplete/unqualified three-seed FPNet cannot predict for ranking')
    if selected_view!=gate['selected_view']:raise ValueError('Fingerprint view differs from M2 freeze')
    sums={s['seed']:s['checkpoint_sha256'] for s in summary['seeds']}
    if set(sums)!=set(SEEDS) or any(s['completed_epochs']<6 for s in summary['seeds']):
        raise ValueError('Incomplete three-seed training')
    ensemble=[];manifest=[];bits=None
    for seed in SEEDS:
        path=models_root/f'fpnet_seed_{seed}.pt'
        if sha256(path)!=sums[seed]:raise ValueError('Selected FPNet checkpoint SHA mismatch')
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        current=np.asarray(checkpoint['bits'],int)
        if bits is not None and not np.array_equal(bits,current):raise ValueError('Seed fingerprint bits differ')
        bits=current
        if checkpoint['preprocessing']!=dataset.preprocessing:raise ValueError('Query preprocessing differs from train freeze')
        raw=FPNet(checkpoint['metadata_dim'],len(bits)).to('cuda:0')
        raw.load_state_dict(checkpoint['state_dict']);raw.eval()
        model=torch.nn.DataParallel(raw) if torch.cuda.device_count()>1 else raw
        logits=np.empty((len(dataset),len(bits)),np.float32)
        loader=DataLoader(dataset,batch_size=64,num_workers=2,pin_memory=True,shuffle=False)
        with torch.inference_mode():
            for batch in loader:
                with torch.autocast('cuda',dtype=torch.float16):
                    score=model(*(batch[k].to('cuda:0') for k in ['peaks','mask','meta']))
                value=score.float().cpu().numpy()
                if not np.isfinite(value).all():raise FloatingPointError('Nonfinite frozen fingerprint prediction')
                logits[batch['index'].numpy()]=value
        single=np.stack([logits[g['single']].mean(0) for g in dataset.layout])
        merged=np.stack([logits[g['merged']].mean(0) for g in dataset.layout])
        ensemble.append(single if selected_view=='single' else .5*single+.5*merged)
        manifest.append({'seed':seed,'checkpoint_sha256':sums[seed],
            'selected_epoch':checkpoint['selected_epoch'],'selected_weight_updates':checkpoint['optimizer_updates']})
        del model,raw,checkpoint,logits;torch.cuda.empty_cache()
        print(json.dumps({'stage':'M3_frozen_fp_prediction','seed':seed,'identities':len(dataset.identities),'views':len(dataset)}),flush=True)
    np.save(output/'logits.npy',np.mean(ensemble,axis=0));np.save(output/'bits.npy',bits)
    write_json(output/'identities.json',dataset.identities)
    write_json(output/'prediction_manifest.json',{'models':manifest,'view':selected_view,
        'identities':len(dataset.identities),'views':len(dataset),'target_arrays_read':False,
        'logits_sha256':sha256(output/'logits.npy')})


def main():
    if sys.argv[1]=='--runtime':
        import pandas as pd
        models,parquet,view,output=sys.argv[2:]
        preprocessing=json.loads((Path(models)/'train_fitted_statistics.json').read_text())['preprocessing']
        frame=pd.read_parquet(parquet)
        required={'molecule_id','record_id','ms2_mzs','ms2_normalized_intensities',
                  'precursor_mz','adduct','ionization_mode','instrument_type','collision_energy_ev'}
        if not required<=set(frame):raise ValueError('Runtime spectrum schema incomplete')
        frozen_predict(models,RuntimeViews(frame.loc[:,sorted(required)],preprocessing),view,output)
        return
    models,cache,key_file,view,output=sys.argv[1:]
    keys=json.loads(Path(key_file).read_text())
    frozen_predict(models,InputViews(cache,keys),view,output)

if __name__=='__main__':main()
