import pandas as pd, numpy as np, time
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*'); print('rdkit',rdBase.rdkitVersion)
import os; D=os.environ.get('CASMI_DATA','data')+'/'
m=pd.read_parquet(D+'train.parquet',columns=['inchikey14','normalized_smiles','ingest_lib'])
u=m.drop_duplicates('inchikey14')
np_lib=u[u.ingest_lib!='enveda-180'].sample(6000,random_state=0)
e=u[u.ingest_lib=='enveda-180'].sample(3000,random_state=0)
te=rdMolStandardize.TautomerEnumerator()
def key(s):
    mol=Chem.MolFromSmiles(s)
    if mol is None: return None
    try: mol=te.Canonicalize(mol)
    except Exception: pass
    k=Chem.MolToInchiKey(mol); return k[:14] if k else None
for name,df in [('non-enveda180',np_lib),('enveda180',e)]:
    t=time.time(); ks=df.normalized_smiles.map(key); dt=time.time()-t
    bad=(ks!=df.inchikey14)
    print(name, 'n',len(df),'metric-key != inchikey14 column:',round(bad.mean(),4),'fail',ks.isna().sum(), f'{dt/len(df)*1000:.1f} ms/struct')
    # collisions: distinct column keys mapping to same metric key
    print('  collapsed groups', len(df)-ks.nunique())
