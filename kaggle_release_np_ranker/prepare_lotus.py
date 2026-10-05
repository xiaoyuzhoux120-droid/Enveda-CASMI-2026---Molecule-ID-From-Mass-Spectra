"""Prepare a full, query-independent official-identity LOTUS asset."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from concurrent.futures import ProcessPoolExecutor
import json
import time
import numpy as np
import pandas as pd
from rdkit import rdBase
from casmi_ml.np_pairtail_structures import canonical_record,sha256


def main():
    if rdBase.rdkitVersion!='2026.03.3':raise ValueError('Pinned identity runtime required')
    source=ROOT/'external/lotus/lotus_structures.parquet'
    if sha256(source)!='a7224f20e88018c50a03982ec579d6d0cf129ecd6fec70744ea7368ad2dbfd05':raise ValueError('LOTUS source mismatch')
    out=ROOT/'kaggle_release_np_ranker/dataset';out.mkdir(exist_ok=True)
    data=pd.read_parquet(source)
    neutral=data.loc[data.formal_charge.eq(0),['normalized_smiles','mass']].copy()
    smiles=sorted(set(neutral.normalized_smiles));mapping={};canonical={};start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for count,(smi,value) in enumerate(zip(smiles,pool.map(canonical_record,smiles,chunksize=32)),1):
            if value is None:raise ValueError('Invalid graph in audited LOTUS asset')
            mapping[smi],canonical[smi]=value
            if count%1000==0:
                print(json.dumps({'stage':'M3_LOTUS_asset_identity_audit','completed':count,'total':len(smiles),'seconds':time.monotonic()-start}),flush=True)
    neutral['identity']=neutral.normalized_smiles.map(mapping)
    neutral['canonical_smiles']=neutral.normalized_smiles.map(canonical)
    neutral['sources']='LOTUS'
    path=out/'lotus_official_pool.parquet';neutral.to_parquet(path,index=False)
    manifest={'source_sha256':sha256(source),'source_version':'LOTUS v11 2026-04-13','source_url':'https://zenodo.org/records/19360665',
        'license':'CC BY 4.0','authors':['Adriano Rutz','Jonathan Bisson','Pierre-Marie Allard'],
        'rdkit_version':rdBase.rdkitVersion,'charged_excluded':int((~data.formal_charge.eq(0)).sum()),
        'rows':len(neutral),'identities':int(neutral.identity.nunique()),'scope':'full source, no query window or label inputs',
        'sha256':sha256(path),'seconds':time.monotonic()-start,'query_labels_used':False}
    (out/'lotus_official_pool.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'kaggle_release_np_ranker/lotus_official_preparation.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
