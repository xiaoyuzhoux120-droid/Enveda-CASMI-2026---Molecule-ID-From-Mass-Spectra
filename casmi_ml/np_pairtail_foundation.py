"""Auditable M0/M1 foundation for the frozen NP-PairTail experiment.

This is a research entry, never a competition entry. Acceptance predictions are
deliberately not computed here. All thresholds are development-only choices.
"""
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem.MolStandardize import rdMolStandardize

from baseline import ADDUCT_MASS, formula_mass, make_matrix, vectorize
from hybrid import blend, coconut_rank, library_rank
from casmi_ml.chemical_priors import candidate_scores, extract_evidence, load_rules, rerank
from casmi_ml.hybrid_chemistry import COCONUT_COLUMNS, HybridChemistry
from casmi_ml.v1_routing import RoutingHybrid, CONFIGS, unique_official_candidates
from casmi_ml.data import write_json

SALT = 'casmi26-np-pairtail-20261004-v1:'
META = ['normalized_smiles', 'inchikey14', 'molecular_formula', 'precursor_mz',
        'adduct', 'ionization_mode', 'instrument_type', 'collision_energy_ev',
        'precursor_error_ppm', 'ingest_lib']


def order(key, namespace='identity'):
    return int.from_bytes(hashlib.sha256((SALT + namespace + ':' + str(key)).encode()).digest()[:8], 'big')


def partition(key):
    bucket = order(key) % 10000
    return 'train' if bucket < 8000 else 'development' if bucket < 9000 else 'acceptance'


def finite(value):
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def reference_mass(row):
    formula = formula_mass(row.get('molecular_formula'))
    precursor = finite(row.get('precursor_mz'))
    offset = ADDUCT_MASS.get(row.get('adduct'))
    reverse = precursor - offset if precursor is not None and offset is not None else math.nan
    mass = formula if math.isfinite(formula) and formula > 0 else reverse
    source = 'formula' if math.isfinite(formula) and formula > 0 else 'precursor_adduct' if math.isfinite(reverse) and reverse > 0 else 'unavailable'
    ppm = (reverse - formula) / formula * 1e6 if math.isfinite(reverse) and math.isfinite(formula) and formula > 0 else None
    return {'mass': mass, 'mass_source': source, 'recomputed_ppm': ppm,
            'adduct_parse_status': 'supported_charge_1' if offset is not None else 'unsupported',
            'provided_ppm': finite(row.get('precursor_error_ppm'))}


def mass_keep(audit, threshold, baseline=False):
    if not math.isfinite(audit['mass']) or audit['mass'] <= 0:
        return False
    ppm = audit['provided_ppm'] if baseline else audit['recomputed_ppm']
    return ppm is None or abs(ppm) <= threshold


def representative_priority(reference, query, instrument=True):
    mode, adduct, family = reference['ionization_mode'], reference['adduct'], reference['instrument_type']
    qmode = {str(x) for x in query.ionization_mode}
    qadduct = {str(x) for x in query.adduct}
    qfamily = {str(x).lower() for x in query.instrument_type}
    return (str(mode) not in qmode, str(adduct) not in qadduct,
            str(family).lower() not in qfamily if instrument else False,
            -reference['valid_peaks'], reference['record_id'])


class Progress:
    def __init__(self, root, total=14400, prior_seconds=0.):
        self.root, self.start, self.total = Path(root), time.monotonic(), total
        self.prior_seconds = float(prior_seconds)
        self.events = []
        self.identity_cache = {}
    def emit(self, stage, **values):
        import resource
        seconds = self.prior_seconds + time.monotonic() - self.start
        row = {'stage': stage, 'elapsed_seconds': seconds,
               'cpu_peak_rss_native': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               **values}
        self.events.append(row)
        write_json(self.root / 'runtime_profile.json', self.events)
        write_json(self.root / 'run_status.json', row)
        print(json.dumps(row, allow_nan=False), flush=True)
        if seconds > self.total:
            raise TimeoutError('Frozen four-hour total research budget exceeded')
    def deadline(self, started, seconds, stage):
        if time.monotonic() - started > seconds:
            raise TimeoutError(f'{stage} exceeded frozen stage budget')


def spectrum_signature(row):
    # Both exact cleaned numerical spectra and the actual retrieval bins are
    # checked. Exclude cross-split copies regardless of molecular annotation.
    mz = np.asarray(row['ms2_mzs'], dtype='<f8')
    intensity = np.asarray(row['ms2_normalized_intensities'], dtype='<f8')
    exact = hashlib.sha256(mz.tobytes() + b'|' + intensity.tobytes()).hexdigest()
    bins = vectorize(mz, intensity)
    numeric = json.dumps(sorted(bins.items()), separators=(',', ':'))
    return exact, hashlib.sha256(numeric.encode()).hexdigest(), bins


def prepare(train_path, root, mapping, observed, protocol, progress):
    started = time.monotonic()
    if rdBase.rdkitVersion != protocol['rdkit_version']:
        raise ValueError('Canonical identity cache RDKit version mismatch')
    frame = pd.read_parquet(train_path, columns=META)
    frame['record_id'] = np.arange(len(frame), dtype=np.int64)
    frame['identity'] = frame.normalized_smiles.map(mapping)
    absent = frame.normalized_smiles.notna() & ~frame.normalized_smiles.isin(mapping)
    if absent.any():
        raise ValueError('Input has structures absent from frozen canonical cache')
    bad = int(frame.identity.isna().sum())
    frame = frame[frame.identity.notna()].copy()
    diagnostics = set(frame.loc[frame.ingest_lib.eq('enveda-np-examples'), 'identity'])
    identities = sorted(set(frame.identity), key=lambda x: (order(x), x))
    selected = {name: [] for name in ('train', 'development', 'acceptance')}
    for identity in identities:
        split = partition(identity)
        if identity in diagnostics or (split == 'acceptance' and identity in observed):
            continue
        selected[split].append(identity)
    selected['train'] = selected['train'][:protocol['train_identity_cap']]
    selected['development'] = selected['development'][:protocol['development_selected']]
    selected['acceptance'] = selected['acceptance'][:protocol['acceptance_selected']]
    owner = {k: split for split, keys in selected.items() for k in keys}
    frame['split'] = frame.identity.map(owner)
    # Deterministic metadata diversity sampling: first of each instrument,
    # adduct, polarity, energy tuple, then the next representative of each.
    sampling = frame[frame.split.notna()].copy()
    sampling['ce_signature'] = sampling.collision_energy_ev.map(lambda v: json.dumps([] if v is None else np.asarray(v).tolist()))
    sampling['stable_order'] = sampling.record_id.map(lambda x: order(x, 'spectrum'))
    diversity = ['identity', 'adduct', 'ionization_mode', 'instrument_type', 'ce_signature']
    sampling = sampling.sort_values('stable_order')
    sampling['diversity_round'] = sampling.groupby(diversity, dropna=False).cumcount()
    sampling = sampling.sort_values(['diversity_round', 'stable_order'])
    sampling['within_identity'] = sampling.groupby('identity').cumcount()
    keep = sampling.within_identity < np.where(sampling.split.eq('train'), 8, 4)
    chosen_rows = set(sampling.loc[keep, 'record_id'])
    progress.emit('M0_identity_split', identities={k: len(v) for k,v in selected.items()},
                  selected_spectra=len(chosen_rows), missing_or_invalid_identity_rows=bad)
    rows, signatures = [], defaultdict(set)
    columns = META + ['ms2_mzs', 'ms2_normalized_intensities']
    offset, empty, last = 0, 0, time.monotonic()
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=8192, columns=columns):
        indexes = np.asarray([i for i in range(len(batch)) if offset+i in chosen_rows], dtype=np.int64)
        selected_batch = batch.take(indexes).to_pandas() if len(indexes) else pd.DataFrame()
        for position, row in zip(indexes, selected_batch.to_dict('records')):
            rid = offset + int(position)
            ident = mapping[row['normalized_smiles']]
            exact, binned, vector = spectrum_signature(row)
            if not vector:
                empty += 1
                continue
            row.update(record_id=rid, identity=ident, split=owner[ident], exact_signature=exact, binned_signature=binned)
            signatures[('exact', exact)].add(owner[ident])
            signatures[('binned', binned)].add(owner[ident])
            rows.append(row)
        offset += len(batch)
        if time.monotonic()-last >= 60:
            progress.emit('M0_spectrum_copy_audit', scanned=offset, total=len(frame)+bad,
                          retained=len(rows), empty_spectra=empty)
            last=time.monotonic()
        progress.deadline(started, 1200, 'M0 data/identity audit')
    cross = {key for key, splits in signatures.items() if len(splits)>1}
    retained = [r for r in rows if ('exact', r['exact_signature']) not in cross and ('binned', r['binned_signature']) not in cross]
    frames = {}
    for split in selected:
        data = pd.DataFrame([r for r in retained if r['split']==split])
        if split != 'train':
            neutral = data.precursor_mz - data.adduct.map(ADDUCT_MASS)
            data = data[neutral.between(157,1159)].copy()
        data.to_parquet(root / f'{split}.parquet', index=False)
        frames[split] = data
    actual = {s:int(f.identity.nunique()) for s,f in frames.items()}
    manifest = {'salt':SALT, 'ratio':[.8,.1,.1], 'selection_limits':{s:len(v) for s,v in selected.items()},
                'identity_lists':{s:sorted(set(f.identity)) for s,f in frames.items()},
                'acceptance_excluded_observed':sorted(observed),
                'identity_intersections':{a+'_'+b:len(set(frames[a].identity)&set(frames[b].identity))
                                         for a,b in [('train','development'),('train','acceptance'),('development','acceptance')]},
                'cross_split_signature_groups_removed':len(cross),'cross_split_rows_removed':len(rows)-len(retained),
                'cross_split_signature_intersections_after_removal':0,
                'empty_spectra':empty,'actual_identities':actual,
                'sample_diversity':'round-robin metadata tuples, hash record ID, train<=8/eval<=4',
                'canonical_cache_source':'V1 actual retained map; exact input and RDKit version verified',
                'all_historical_queries_recoverable':False}
    write_json(root/'split_manifest.json',manifest)
    if any(manifest['identity_intersections'].values()):
        raise AssertionError('Identity leakage')
    if actual['development']<1000 or actual['acceptance']<1500:
        raise ValueError('Insufficient new development/acceptance sample sizes')
    progress.emit('M0_data_prepared', actual_identities=actual, acceptance_predictions_computed=False)
    return frames


def restore_prepared(directory, root, mapping, observed, protocol, progress):
    """Resume the completed M0 checkpoint; independently verify its invariants."""
    import shutil
    directory=Path(directory);root=Path(root)
    stored=json.loads((directory/'protocol_np_pairtail_20261004.json').read_text())
    if stored!=protocol:
        raise ValueError('Cached M0 scientific protocol differs')
    manifest=json.loads((directory/'split_manifest.json').read_text())
    frames={};signatures={};hashes={}
    for split in ['train','development','acceptance']:
        path=directory/f'{split}.parquet'
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
        hashes[path.name]=digest.hexdigest()
        frame=pd.read_parquet(path)
        actual=set(frame.identity)
        if actual!=set(manifest['identity_lists'][split]):
            raise ValueError('Cached M0 identity membership differs')
        if not (frame.normalized_smiles.map(mapping)==frame.identity).all():
            raise ValueError('Cached M0 aliases differ from canonical map')
        if any(partition(identity)!=split for identity in actual):
            raise ValueError('Cached M0 split salt differs')
        if split=='acceptance' and actual & observed:
            raise ValueError('Cached acceptance includes observed identities')
        if frame.ingest_lib.eq('enveda-np-examples').any():
            raise ValueError('Diagnostic-source query in cached main cohorts')
        if frame.groupby('identity').size().max()>protocol['spectra_per_identity'][split]:
            raise ValueError('Cached M0 spectra-per-identity cap exceeded')
        signatures[split]={key:set(frame[key]) for key in ['exact_signature','binned_signature']}
        frames[split]=frame
        shutil.copyfile(path,root/path.name)
    for a,b in [('train','development'),('train','acceptance'),('development','acceptance')]:
        if set(frames[a].identity)&set(frames[b].identity):raise ValueError('Cached M0 identity overlap')
        for kind in signatures[a]:
            if signatures[a][kind]&signatures[b][kind]:raise ValueError('Cached M0 numerical spectrum overlap')
    if frames['development'].identity.nunique()<1000 or frames['acceptance'].identity.nunique()<1500:
        raise ValueError('Cached M0 insufficient sample sizes')
    shutil.copyfile(directory/'split_manifest.json',root/'split_manifest.json')
    write_json(root/'M0_restored_checkpoint.json',{'source':str(directory),'sha256':hashes,
        'identity_membership_and_signatures_rechecked':True,'acceptance_predictions_computed':False})
    progress.emit('M0_checkpoint_restored',actual_identities={s:int(f.identity.nunique()) for s,f in frames.items()},
        acceptance_predictions_computed=False,checkpoint_hashes=hashes)
    return frames


def load_references(train_path, masses, query_frames, mapping, root, progress):
    started=time.monotonic(); target=np.sort(masses)
    query=pd.concat(query_frames,ignore_index=True)
    query_signatures=set(query.binned_signature)
    query_ids=set(query.identity)
    records, metadata, audit_rows = [], [], []
    counters=Counter(); offset=0; last=time.monotonic()
    for batch in pq.ParquetFile(train_path).iter_batches(batch_size=8192,columns=META+['ms2_mzs','ms2_normalized_intensities']):
        for i,row in enumerate(batch.to_pandas().to_dict('records')):
            info=reference_mass(row); counters['rows']+=1;counters['mass_'+info['mass_source']]+=1
            counters['formula_parse_failure']+=int(not math.isfinite(formula_mass(row['molecular_formula'])))
            if info['recomputed_ppm'] is not None:
                counters['recomputed_gt30']+=int(abs(info['recomputed_ppm'])>30)
                for limit in [10,20,30]:
                    counters[f'recomputed_gt{limit}_ppm']+=int(abs(info['recomputed_ppm'])>limit)
                counters['old_gt30_new_le30']+=int(info['provided_ppm'] is not None and abs(info['provided_ppm'])>30 and abs(info['recomputed_ppm'])<=30)
            mass=info['mass']
            if not math.isfinite(mass) or mass<=0:continue
            j=np.searchsorted(target,mass);distance=min(abs(mass-target[max(0,j-1)]),abs(mass-target[min(j,len(target)-1)]))
            if distance>mass*35e-6:continue
            exact,binned,vector=spectrum_signature(row)
            if not vector or binned in query_signatures:continue
            ident=mapping.get(row['normalized_smiles'])
            if not ident:continue
            metadata.append({**info,'record_id':offset+i,'identity':ident,'adduct':row['adduct'],
                             'ionization_mode':row['ionization_mode'],'instrument_type':row['instrument_type'],
                             'valid_peaks':len(vector),'is_query_identity':ident in query_ids})
            records.append((mass,str(row['inchikey14']),row['normalized_smiles'],vector))
            audit_rows.append({'record_id':offset+i,**info,'identity':ident})
        offset+=len(batch)
        if time.monotonic()-last>=60:
            progress.emit('M1_reference_scan',scanned=offset,retained=len(records),counters=dict(counters));last=time.monotonic()
        progress.deadline(started,1500,'M1 AFIX/reference index')
    pd.DataFrame(audit_rows).to_parquet(root/'reference_mass_audit.parquet',index=False)
    write_json(root/'reference_mass_summary.json',dict(counters))
    if counters['formula_parse_failure']/max(1,counters['rows'])>.01:
        raise ValueError('Formula parsing failure >1%; PRD M1 stop')
    if not records:raise ValueError('No reference records')
    progress.emit('M1_reference_ready',references=len(records),counters=dict(counters))
    return records,metadata


class SharedAnalog:
    """Memoize pure analog scores with complete scoring-input keys.

    Includes exact center, catalog signature, and all 25 weighted neighbors.
    It never reads query identity, ground truth, query ID, or a previous CSV.
    """
    def __init__(self):
        self.fingerprints, self.results = {}, {}
        self.hits = self.misses = 0

    def rank(self, engine, center, library, expanded=False):
        pool = engine.unified if expanded else engine.coconut
        masses = engine.new_mass if expanded else engine.old_mass
        order = engine.new_order if expanded else engine.old_order
        # All engines share the same frozen pool; assert that equality once
        # per materialized pool rather than trusting its row count.
        if not hasattr(engine, '_pool_signatures'):
            engine._pool_signatures = tuple(hashlib.sha256(
                pd.util.hash_pandas_object(p[COCONUT_COLUMNS], index=True).to_numpy().tobytes()).hexdigest()
                for p in [engine.coconut, engine.unified])
        signature = engine._pool_signatures[int(expanded)]
        neighbors = tuple((key, smiles, float(score).hex()) for key, smiles, score in library[:25])
        cache_key = (signature, float(center).hex(), neighbors)
        if cache_key not in self.results:
            self.results[cache_key] = coconut_rank(center, library, pool, masses, order, self.fingerprints)
            self.misses += 1
        else:
            self.hits += 1
        return self.results[cache_key]


class CachedRoutingHybrid(RoutingHybrid):
    def __init__(self, *args, shared=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared = shared or SharedAnalog()

    def controls(self, group, configs=CONFIGS):
        if len(configs) != 1 or configs[0] != CONFIGS[0]:
            raise ValueError('Foundation memoization is restricted to frozen C0 routing')
        started = time.monotonic()
        center, library = library_rank(group, self.records, self.matrix, self.masses, self.order)
        old = self.shared.rank(self, center, library)
        pairs = blend(library, old)
        confidence = float(library[0][2]) if library else 0.
        margin = confidence-float(library[1][2]) if len(library)>1 else confidence
        evidence = []
        if confidence < .5:
            analog = self.shared.rank(self, center, library, True)
            pairs = blend(library, analog)
            evidence = extract_evidence(group, self.rules)
            if evidence:
                structures = dict(pairs); scores, _ = candidate_scores(structures, evidence, self.rules)
                pairs = [(key, structures[key]) for key in rerank([k for k,_ in pairs], scores, self.weight)]
        return {'B0':pairs}, {'confidence':confidence, 'margin':margin,
            'protected':{'B0':confidence >= .5}, 'rule_matches':len(evidence),
            'seconds':time.monotonic()-started, 'analog_cache_hits':self.shared.hits,
            'analog_cache_misses':self.shared.misses}


class RepresentativeHybrid(CachedRoutingHybrid):
    def __init__(self,records,meta,coconut,catalog,rules,instrument,shared=None):
        super().__init__(records,coconut,catalog,rules,shared=shared)
        self.meta,self.instrument=meta,instrument
    def controls(self,group,configs=CONFIGS):
        # Stable metadata representative of each structure in the same mass
        # window used by V1. No truth, query ID or candidate score in selection.
        center=float(np.median(group.precursor_mz-group.adduct.map(ADDUCT_MASS)))
        delta=max(center*35e-6,.006);sorted_mass=self.masses[self.order]
        ids=self.order[np.searchsorted(sorted_mass,center-delta):np.searchsorted(sorted_mass,center+delta,side='right')]
        if not len(ids):
            at=np.searchsorted(sorted_mass,center);ids=self.order[max(0,at-150):at+150]
        chosen={}
        for i in ids:
            key=self.meta[i]['identity'];priority=representative_priority(self.meta[i],group,self.instrument)
            if key not in chosen or priority<chosen[key][0]:chosen[key]=(priority,i)
        index=sorted(i for _,i in chosen.values())
        records=[self.records[i] for i in index]
        mass=np.asarray([r[0] for r in records]);matrix=self.matrix[index]
        center,library=library_rank(group,records,matrix,mass,np.argsort(mass))
        confidence=library[0][2] if library else 0.
        old=self.shared.rank(self,center,library)
        output=blend(library,old);evidence=[]
        if confidence<.5:
            analog=self.shared.rank(self,center,library,True)
            output=blend(library,analog);evidence=extract_evidence(group,self.rules)
            if evidence:
                structures=dict(output);scores,_=candidate_scores(structures,evidence,self.rules)
                output=[(k,structures[k]) for k in rerank(list(structures),scores,self.weight)]
        return {'B0':output},{'confidence':float(confidence),'representatives':len(records),'instrument_aware':self.instrument,
            'analog_cache_hits':self.shared.hits,'analog_cache_misses':self.shared.misses}


def evaluate(engine,frame,mapping,progress,label,parity=False):
    enumerator=rdMolStandardize.TautomerEnumerator();cache=progress.identity_cache;cases=[];started=time.monotonic()
    for n,(truth,group) in enumerate(frame.groupby('identity',sort=True),1):
        outputs,audit=engine.controls(group,[CONFIGS[0]])
        if parity and n<=100:
            original=HybridChemistry.__new__(HybridChemistry);original.__dict__=engine.__dict__
            expected=original.variants(group,{'v1':(True,True)})['v1'][0]
            if outputs['B0']!=expected:raise AssertionError('C0 frozen V1 parity failed')
        pairs=unique_official_candidates(outputs['B0'],cache,enumerator)
        ids=[cache[s] for _,s in pairs];rank=ids.index(truth)+1 if truth in ids else 0
        cases.append({'identity':truth,'rr':1/rank if rank else 0.,'rank':rank,
                      'ranking':[{'key':k,'smiles':s,'identity':cache[s]} for k,s in pairs],**audit})
        if n%25==0:
            seconds=time.monotonic()-started;total=int(frame.identity.nunique())
            progress.emit(label,completed=n,total=total,seconds=seconds,
                queries_per_second=n/max(seconds,1e-6),estimated_remaining_seconds=(total-n)*seconds/n,
                analog_cache_hits=getattr(getattr(engine,'shared',None),'hits',0),
                analog_cache_misses=getattr(getattr(engine,'shared',None),'misses',0))
        if n==100 and (time.monotonic()-started)/100>6:
            raise TimeoutError('100-query pilot exceeds six seconds/query')
    write_json(progress.root/(label+'_cases.json'),cases)
    return {'molecules':len(cases),'mrr25':float(np.mean([r['rr'] for r in cases])),
            'top1':float(np.mean([r['rank']==1 for r in cases])),
            'recall25':float(np.mean([r['rank']>0 for r in cases]))}


def run(train_path,coconut_path,catalog_path,dictionary_path,cache_path,observed_path,protocol,root,prior_seconds=0.,prepared=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);progress=Progress(root,prior_seconds=prior_seconds)
    write_json(root/'protocol_np_pairtail_20261004.json',protocol)
    mapping=json.loads(Path(cache_path).read_text());observed=set(json.loads(Path(observed_path).read_text()))
    try:
        frames=restore_prepared(prepared,root,mapping,observed,protocol,progress) if prepared else prepare(train_path,root,mapping,observed,protocol,progress)
        # Only development references/queries are used by this stage.
        dev=frames['development'];masses=(dev.precursor_mz-dev.adduct.map(ADDUCT_MASS)).to_numpy()
        records,meta=load_references(train_path,masses,[dev],mapping,root,progress)
        coconut=pd.read_parquet(coconut_path,columns=COCONUT_COLUMNS);catalog=pd.read_parquet(catalog_path);rules=load_rules(dictionary_path)
        results={};shared=SharedAnalog()
        known_baseline_identities={m['identity'] for m in meta if mass_keep(m,30,True) and m['mass_source']=='formula'}
        for condition in ['unknown','known']:
            include=[i for i,m in enumerate(meta) if condition=='known' or not m['is_query_identity']]
            baseline_ids=[i for i in include if mass_keep(meta[i],30,True) and meta[i]['mass_source']=='formula']
            remaining={meta[i]['identity'] for i in baseline_ids}
            query=dev if condition=='unknown' else dev[dev.identity.isin(remaining)]
            engine=CachedRoutingHybrid([records[i] for i in baseline_ids],coconut,catalog,rules,shared=shared)
            results['C0_'+condition]=evaluate(engine,query,mapping,progress,'C0_'+condition,parity=True)
            for threshold in [10,20,30]:
                ids=[i for i in include if mass_keep(meta[i],threshold)]
                engine=CachedRoutingHybrid([records[i] for i in ids],coconut,catalog,rules,shared=shared)
                results[f'C1_ppm{threshold}_{condition}']=evaluate(engine,query,mapping,progress,f'C1_ppm{threshold}_{condition}')
        valid=[t for t in [10,20,30] if results[f'C1_ppm{t}_known']['mrr25']>=results['C0_known']['mrr25']-.001 and results[f'C1_ppm{t}_known']['top1']>=results['C0_known']['top1']-.002]
        threshold=max(valid,key=lambda t:(results[f'C1_ppm{t}_unknown']['mrr25'],-[10,20,30].index(t))) if valid else None
        write_json(root/'M1_afix_selection.json',{'threshold_ppm':threshold,'selected_on':'development only','acceptance_opened':False})
        if threshold is not None:
            for condition in ['unknown','known']:
                ids=[i for i,m in enumerate(meta) if (condition=='known' or not m['is_query_identity']) and mass_keep(m,threshold)]
                query=dev if condition=='unknown' else dev[dev.identity.isin(known_baseline_identities)]
                for aware in [False,True]:
                    label=f'C2_instrument{int(aware)}_'+condition
                    engine=RepresentativeHybrid([records[i] for i in ids],[meta[i] for i in ids],coconut,catalog,rules,aware,shared=shared)
                    results[label]=evaluate(engine,query,mapping,progress,label)
        write_json(root/'development_foundation_results.json',results)
        progress.emit('M1_foundation_complete',afix_threshold=threshold,acceptance_opened=False,
                      next_stage='M2 real three-seed FPNet; not yet trained',competition_submission_allowed=False)
        return results
    except Exception as exc:
        progress.emit('foundation_failed',error_type=type(exc).__name__,error=str(exc),
                      acceptance_opened=False,competition_submission_allowed=False)
        raise
