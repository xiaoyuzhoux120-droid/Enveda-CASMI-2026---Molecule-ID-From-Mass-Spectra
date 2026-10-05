"""Query-independent runtime indexes for M3 and later frozen inference."""
import json
import math
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from baseline import ADDUCT_MASS,make_matrix,vectorize
from hybrid import library_rank
from casmi_ml.np_pairtail_foundation import (META,CachedRoutingHybrid,Progress,
    reference_mass,mass_keep,spectrum_signature)
from casmi_ml.v1_routing import CONFIGS,unique_official_candidates
from casmi_ml.np_pairtail_structures import canonical_target


def center_mass(group):
    values=(group.precursor_mz-group.adduct.map(ADDUCT_MASS)).to_numpy(float)
    if not len(values) or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Unsupported charge-1 adduct/neutral query mass')
    return float(np.median(values))


def _in_scope(mass,targets):
    if not len(targets):return False
    pos=np.searchsorted(targets,mass)
    distance=min(abs(mass-targets[max(pos-1,0)]),abs(mass-targets[min(pos,len(targets)-1)]))
    return bool(distance<=mass*35e-6)


def reference_scan(train_path,query_frames,mapping,root,progress,deadline):
    scope_centers={name:np.sort(np.array([center_mass(group) for _,group in f.groupby('identity',sort=True)]))
                   for name,f in query_frames.items()}
    centers=np.sort(np.concatenate(list(scope_centers.values())))
    if not len(centers):raise ValueError('Empty reference scope')
    signatures={name:set(frame.binned_signature) for name,frame in query_frames.items()}
    records,metadata=[],[];offset=0;last=time.monotonic()
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=8192,columns=META+['ms2_mzs','ms2_normalized_intensities']):
        for local,row in enumerate(batch.to_pandas().to_dict('records')):
            audit=reference_mass(row);mass=audit['mass']
            if not math.isfinite(mass) or mass<=0:continue
            pos=np.searchsorted(centers,mass)
            distance=min(abs(mass-centers[max(pos-1,0)]),abs(mass-centers[min(pos,len(centers)-1)]))
            # Exact original foundation scan scope; query retrieval retains its
            # own .006 Da floor. No arbitrary union-wide mass expansion.
            if distance>mass*35e-6:continue
            ident=mapping.get(row['normalized_smiles'])
            if not ident:raise ValueError('Reference structure missing frozen official identity map')
            _,signature,vector=spectrum_signature(row)
            if not vector:continue
            records.append((mass,str(row['inchikey14']),row['normalized_smiles'],vector))
            metadata.append({**audit,'record_id':offset+local,'identity':ident,
                'adduct':row['adduct'],'ionization_mode':row['ionization_mode'],
                'instrument_type':row['instrument_type'],'valid_peaks':len(vector),
                **{'is_'+name+'_query_signature':signature in sigs for name,sigs in signatures.items()},
                **{'in_'+name+'_mass_scope':_in_scope(mass,targets) for name,targets in scope_centers.items()}})
        offset+=len(batch)
        if time.monotonic()>=deadline:raise TimeoutError('Reference construction research/module budget exhausted')
        if time.monotonic()-last>=60:
            progress.emit('M3_reference_scan',scanned=offset,retained=len(records));last=time.monotonic()
    if not records:raise ValueError('No retained reference spectra')
    pd.DataFrame(metadata).to_parquet(Path(root)/'reference_metadata.parquet',index=False)
    sparse.save_npz(Path(root)/'reference_spectra.npz',make_matrix([r[3] for r in records]))
    pd.DataFrame({'mass':[r[0] for r in records],'raw_key':[r[1] for r in records],
        'normalized_smiles':[r[2] for r in records]}).to_parquet(Path(root)/'reference_rows.parquet',index=False)
    return records,metadata


class RuntimeRetrieval:
    def __init__(self,records,metadata,coconut,catalog,rules,mapping):
        self.engine=CachedRoutingHybrid(records,coconut,catalog,rules)
        self.meta=metadata;self.mapping=mapping;self.rules=rules
        self.identity_cache={};self.enum=rdMolStandardize.TautomerEnumerator()

    def score(self,group):
        engine=self.engine
        center,raw_library=library_rank(group,engine.records,engine.matrix,engine.masses,engine.order)
        outputs,audit=engine.controls(group,[CONFIGS[0]])
        pairs=unique_official_candidates(outputs['B0'],self.identity_cache,self.enum)
        baseline=[{'identity':self.identity_cache[s],'normalized_smiles':s,'mass':float(rdMolDescriptors.CalcExactMolWt(Chem.MolFromSmiles(s))),
                   'sources':'train/library' if k in {r[1] for r in raw_library} else 'COCONUT'} for k,s in pairs]
        canonical_library={}
        for raw_key,smi,score in raw_library:
            key=self.mapping.get(smi)
            if key is None:
                # Public proposal aliases are normalized by the official target
                # helper; this fallback is independent of any query label.
                from casmi_ml.np_pairtail_structures import canonical_record
                value=canonical_record(smi)
                if value is None:raise ValueError('Invalid reference structure')
                key=value[0];self.mapping[smi]=key
            canonical_library.setdefault(key,(key,smi,float(score)))
            if score>canonical_library[key][2]:canonical_library[key]=(key,smi,float(score))
        library=sorted(canonical_library.values(),key=lambda r:(-r[2],r[0]))
        delta=max(center*35e-6,.006);sorted_mass=engine.masses[engine.order]
        ids=engine.order[np.searchsorted(sorted_mass,center-delta):np.searchsorted(sorted_mass,center+delta,side='right')]
        if not len(ids):
            pos=np.searchsorted(sorted_mass,center);ids=engine.order[max(0,pos-150):pos+150]
        vectors=[vectorize(r.ms2_mzs,r.ms2_normalized_intensities) for r in group.itertuples()]
        scores=(make_matrix(vectors)@engine.matrix[ids].T).toarray()
        by_key=defaultdict(list)
        for col,index in enumerate(ids):by_key[self.meta[index]['identity']].append((col,index))
        info={}
        query_adducts=set(group.adduct);instruments=set(group.instrument_type.astype(str).str.lower())
        for key,columns in by_key.items():
            cols=[p[0] for p in columns];indexes=[p[1] for p in columns]
            per_query=scores[:,cols].max(1)
            shifted=[]
            # Actual precursor/adduct-adjusted bin shifts for retrieved spectra,
            # not a synthetic forward spectrum or domain-unknown zero score.
            for query,v in zip(group.itertuples(),vectors):
                neutral_query=query.precursor_mz-ADDUCT_MASS[query.adduct]
                best=0.
                for index in indexes:
                    meta=self.meta[index];ref_offset=ADDUCT_MASS.get(meta['adduct'])
                    if ref_offset is None:continue
                    shift=(neutral_query-meta['mass'])+(ADDUCT_MASS[query.adduct]-ref_offset)
                    bins=int(np.rint(10*shift))
                    best=max(best,sum(weight*v.get(bin_id+bins,0.) for bin_id,weight in engine.records[index][3].items()))
                shifted.append(best)
            info[key]={'spectra':len(indexes),'multispectrum_mean':float(per_query.mean()),
                'shifted_similarity':max(shifted,default=0.),
                'adduct_match':float(np.mean([self.meta[i]['adduct'] in query_adducts for i in indexes])),
                'instrument_match':float(np.mean([str(self.meta[i]['instrument_type']).lower() in instruments for i in indexes]))}
        return baseline,library,info,audit


def build_retrieval(records,metadata,coconut,catalog,rules,mapping,query_frame,signature_scope,
                    condition,threshold,training_identities=None):
    query_identities=set(query_frame.identity)
    ids=[]
    for i,meta in enumerate(metadata):
        if not meta['in_'+signature_scope+'_mass_scope'] or meta['is_'+signature_scope+'_query_signature']:continue
        if training_identities is not None and meta['identity'] not in training_identities:continue
        if condition=='unknown' and meta['identity'] in query_identities:continue
        if threshold is None:
            if not (mass_keep(meta,30,True) and meta['mass_source']=='formula'):continue
        elif not mass_keep(meta,threshold):continue
        ids.append(i)
    return RuntimeRetrieval([records[i] for i in ids],[metadata[i] for i in ids],coconut,catalog,rules,mapping)


def candidate_union(rows,baseline,index):
    # Retain deployment fallback library candidates even outside an empty exact
    # structure window, keeping source flags and the original representative.
    lookup={r['identity']:r for r in rows.to_dict('records')}
    for row in baseline:
        if row['identity'] not in lookup:
            copy=dict(row)
            if row['identity'] in index.sources:copy['sources']=';'.join(sorted(index.sources[row['identity']]))
            lookup[row['identity']]=copy
    return pd.DataFrame([lookup[k] for k in sorted(lookup)],columns=['identity','normalized_smiles','mass','sources'])


def prediction_metrics(cases):
    if not cases:raise ValueError('Empty evaluation cohort')
    ranks=[c['rank'] for c in cases]
    return {'molecules':len(cases),'mrr25':float(np.mean([1/r if r else 0. for r in ranks])),
            'top1':float(np.mean([r==1 for r in ranks])), 'recall25':float(np.mean([r>0 for r in ranks]))}


def evaluate_order(truth,ranking):
    # Sole point where a query label enters final ranking evaluation.
    top=ranking[:25];rank=top.index(truth)+1 if truth in top else 0
    return {'identity':truth,'rank':rank,'rr':1/rank if rank else 0.,
        'top1':int(rank==1),'recall25':int(rank>0),'ranking':top}
