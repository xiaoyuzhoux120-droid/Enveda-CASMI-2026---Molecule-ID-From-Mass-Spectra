import pandas as pd, numpy as np, pyarrow.parquet as pq, json, sys, hashlib
import os; D=os.environ.get('CASMI_DATA','data')+'/'
pf=pq.ParquetFile(D+'train.parquet'); print('train rows',pf.metadata.num_rows); print(pf.schema_arrow.names)
meta=pd.read_parquet(D+'train.parquet',columns=['inchikey14','ingest_lib','adduct','instrument_type','precursor_mz','precursor_error_ppm','num_peaks','molecular_formula','base_peak_intensity'])
test=pd.read_parquet(D+'test.parquet'); print('test',test.shape, test.molecule_id.nunique()); print(test.columns.tolist())
print(test.adduct.value_counts().to_dict()); print(test.ionization_mode.value_counts().to_dict())
print('spectra/molecule', test.groupby('molecule_id').size().describe().to_dict())
out={}
src=meta.groupby('ingest_lib').agg(spectra=('inchikey14','size'),structs=('inchikey14','nunique'))
print(src)
# structure overlap per source
keys_by=meta.groupby('ingest_lib').inchikey14.apply(set)
np_keys=keys_by['enveda-np-examples']
for k,v in keys_by.items(): print(k,'overlap with np-examples',len(v&np_keys))
e180=keys_by['enveda-180']; others=set().union(*[v for k,v in keys_by.items() if k not in('enveda-180',)])
allk=set(meta.inchikey14.dropna())
only180=e180-set().union(*[v for k,v in keys_by.items() if k!='enveda-180'])
print('unique structs',len(allk),'only in enveda-180',len(only180), f'{len(only180)/len(allk):.3f}')
# repo split composition
def dig(k): return int.from_bytes(hashlib.sha256(('42:'+str(k)).encode()).digest()[:8],'big')%10000
cat=pd.DataFrame({'k':list(allk)}); cat['v']=cat.k.map(dig); cat['split']=np.where(cat.v<8000,'train',np.where(cat.v<9000,'dev','holdout'))
cat.loc[cat.k.isin(np_keys),'split']='diagnostic'
cat['only180']=cat.k.isin(only180)
print(cat.groupby('split').only180.agg(['size','mean']))
# precursor error distribution by source
pe=meta.precursor_error_ppm.abs()
print('abs ppm quantiles by lib'); print(meta.assign(a=pe).groupby('ingest_lib').a.quantile([.5,.9,.99]).unstack().round(2))
print('frac >30ppm by lib'); print(meta.assign(a=pe>30).groupby('ingest_lib').a.mean().round(4))
# visible test vs train: match on precursor_mz rounded + num peaks
test['np']=test.ms2_mzs.map(len)
t=meta[['precursor_mz','num_peaks','inchikey14','ingest_lib']].copy(); t['pm']=t.precursor_mz.round(5)
test['pm']=test.precursor_mz.round(5)
m=test.merge(t,left_on=['pm','np'],right_on=['pm','num_peaks'],how='left')
hit=m.groupby('spectrum_id').inchikey14.apply(lambda s:s.notna().any())
print('visible test spectra with train match (precursor_mz 5dp + num_peaks):',hit.mean())
print('matched libs', m.dropna(subset=['inchikey14']).drop_duplicates('spectrum_id').ingest_lib.value_counts().to_dict())
# timsTOF train adducts
tt=meta[meta.ingest_lib.isin(['enveda-180','enveda-np-examples'])]
print('timsTOF train adducts', tt.groupby('ingest_lib').adduct.value_counts().groupby(level=0).head(8).to_dict())
