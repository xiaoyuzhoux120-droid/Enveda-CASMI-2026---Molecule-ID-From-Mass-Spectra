"""Sample a molecule-disjoint [M+H]+/[M-H]- subset of train.parquet for neutral-loss / chemical-prior experiments.

Split is by sha256(inchikey14) so it is reproducible and leaves whole molecules on one side:
  bucket < 8500 -> 'fit' (model training / loss mining), bucket >= 9500 -> 'eval' (isomer-ranking queries, only
  molecules with a non-enveda-180 spectrum, i.e. not the synthetic drug-like library), plus 'np' = all
  enveda-np-examples spectra (organisers' natural-product examples; their structures are excluded from 'fit').
Up to PER_MOL spectra per molecule are kept (first seen in file order, deterministic).
Also writes a structure catalog (one row per inchikey14) used to build same-formula isomer panels.

Usage: python prep.py DATA_DIR OUT_DIR [max_fit_molecules] [max_eval_molecules]
"""
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ADDUCTS = {'[M+H]+', '[M-H]-'}
PER_MOL = 3
COLS = ['inchikey14', 'normalized_smiles', 'molecular_formula', 'ingest_lib', 'instrument_type', 'adduct',
        'precursor_mz', 'ms2_mzs', 'ms2_normalized_intensities', 'collision_energy_ev']


def bucket(key):
    return int.from_bytes(hashlib.sha256(('nl:' + key).encode()).digest()[:4], 'big') % 10000


def main(data_dir, out_dir, max_fit=60000, max_eval=3000):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / 'train.parquet'
    cat = pd.read_parquet(path, columns=['inchikey14', 'normalized_smiles', 'molecular_formula'])
    cat = cat.drop_duplicates('inchikey14')
    cat = cat[cat.inchikey14.str.len() == 14].reset_index(drop=True)
    cat['bucket'] = cat.inchikey14.map(bucket)
    cat.to_parquet(out_dir / 'catalog.parquet', index=False)
    libs = pd.read_parquet(path, columns=['inchikey14', 'ingest_lib'])
    np_keys = set(libs.loc[libs.ingest_lib == 'enveda-np-examples', 'inchikey14'])
    # molecules that have at least one spectrum outside the synthetic, drug-like enveda-180 library
    natural = set(libs.loc[~libs.ingest_lib.isin(['enveda-180', 'enveda-np-examples']), 'inchikey14'])
    del libs
    fit_pool = cat[(cat.bucket < 8500) & ~cat.inchikey14.isin(np_keys)].sort_values('bucket')
    fit_keys = set(fit_pool.inchikey14.head(max_fit))
    # eval molecules: need at least one same-formula isomer in the catalog to form a ranking panel,
    # and a non-enveda-180 spectrum (the test set is natural products, enveda-180 is synthetic/drug-like)
    nform = cat.groupby('molecular_formula').inchikey14.transform('size')
    ev = cat[(cat.bucket >= 9500) & (nform >= 2) & cat.inchikey14.isin(natural) & ~cat.inchikey14.isin(np_keys)]
    eval_keys = set(ev.sort_values('bucket').inchikey14.head(max_eval * 2))
    rows, count = [], defaultdict(int)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=65536, columns=COLS):
        f = batch.to_pandas()
        f = f[f.adduct.isin(ADDUCTS)]
        # enveda-np-examples: the organisers' natural-product examples -> separate 'np' evaluation set (all spectra)
        f = f[f.inchikey14.isin(fit_keys) | (f.inchikey14.isin(eval_keys) & (f.ingest_lib != 'enveda-180'))
              | (f.ingest_lib == 'enveda-np-examples')]
        for r in f.itertuples(index=False):
            if r.ms2_mzs is None or len(r.ms2_mzs) < 3:
                continue
            if r.ingest_lib != 'enveda-np-examples':
                if count[r.inchikey14] >= PER_MOL:
                    continue
                count[r.inchikey14] += 1
            rows.append(r._asdict())
    df = pd.DataFrame(rows)
    df['split'] = np.where(df.ingest_lib == 'enveda-np-examples', 'np',
                           np.where(df.inchikey14.isin(fit_keys), 'fit', 'eval'))
    keep_eval = df.loc[df.split == 'eval'].drop_duplicates('inchikey14').inchikey14.head(max_eval)
    df = df[(df.split != 'eval') | df.inchikey14.isin(set(keep_eval))]
    df.to_parquet(out_dir / 'spectra.parquet', index=False)
    print(df.groupby('split').agg(spectra=('inchikey14', 'size'), molecules=('inchikey14', 'nunique')))
    print(df.groupby(['split', 'adduct']).size())


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], *(int(x) for x in a[2:]))
