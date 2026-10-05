"""M2 fingerprint training on the sealed NP-PairTail foundation outputs.

The acceptance parquet is never opened here. Training samples identities
uniformly and fits vocabularies, target bits and weights on training only.
"""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from torch import nn
from torch.utils.data import DataLoader, Dataset

from casmi_ml.data import CATEGORIES, clean_spectrum, fit_preprocessing, metadata, write_json

SEEDS = (20261004, 20261005, 20261006)
MAX_PEAKS = 64
POS_WEIGHT_CAP = 10.


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_target(smiles, expected_identity, enumerator=None):
    enumerator = enumerator or rdMolStandardize.TautomerEnumerator()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid training target structure')
    mol = enumerator.Canonicalize(mol)
    identity = Chem.MolToInchiKey(mol)[:14]
    if identity != expected_identity:
        raise ValueError('Canonical target identity differs from frozen split')
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return Chem.MolToSmiles(mol), fp.GetFingerprintAsNumPy(mol)


def token_features(mz, weights, losses):
    # Stable top-intensity ordering; losses are the original precursor minus
    # fragment for each peak, including in a merged mixed-adduct view.
    ids = np.argsort(-weights, kind='stable')[:MAX_PEAKS]
    peaks = np.zeros((MAX_PEAKS, 19), np.float32)
    mask = np.zeros(MAX_PEAKS, bool)
    if len(ids):
        m, w, loss = mz[ids], weights[ids], losses[ids]
        frequencies = np.asarray([.01, .1, 1., 10.], np.float32)
        am, al = m[:, None] * frequencies, loss[:, None] * frequencies
        peaks[:len(ids)] = np.concatenate([m[:, None]/1250, w[:, None], loss[:, None]/1250,
            np.sin(am), np.cos(am), np.sin(al), np.cos(al)], axis=1)
        mask[:len(ids)] = True
    if not mask.any():
        mask[0] = True
    return peaks, mask


def single_view(row, preprocessing):
    mz, weights = clean_spectrum(row)
    peaks, mask = token_features(mz, weights, float(row['precursor_mz'])-mz)
    meta = np.concatenate([metadata(row, preprocessing), np.array([0., math.log(2.)], np.float32)])
    return peaks, mask, meta


def merged_views(rows, preprocessing):
    """One max-intensity merged view per polarity, with real per-peak losses.

    Peaks within 0.001 Da share a bin. The strongest normalized original peak
    provides its m/z and precursor-minus-fragment loss. Ties retain lower stable
    record ID. Mixed adduct/instrument metadata is missing, never guessed.
    """
    output = []
    for polarity in sorted({str(r['ionization_mode']) for r in rows}):
        selected = sorted((r for r in rows if str(r['ionization_mode']) == polarity),
                          key=lambda r: int(r['record_id']))
        bins = {}
        for row in selected:
            mz, weights = clean_spectrum(row)
            for m, w in zip(mz, weights):
                key = int(np.rint(float(m)*1000))
                if key not in bins or w > bins[key][1]:
                    bins[key] = (float(m), float(w), float(row['precursor_mz'])-float(m))
        values = np.array([bins[k] for k in sorted(bins)], np.float32).reshape(-1, 3)
        merged = dict(selected[0])
        for category in CATEGORIES:
            options = {str(r.get(category)) for r in selected}
            merged[category] = next(iter(options)) if len(options) == 1 else '<missing>'
        merged['precursor_mz'] = float(np.mean([float(r['precursor_mz']) for r in selected]))
        merged['collision_energy_ev'] = np.concatenate([
            np.asarray([] if r['collision_energy_ev'] is None else r['collision_energy_ev'], float)
            for r in selected])
        peaks, mask = token_features(values[:, 0], values[:, 1], values[:, 2])
        meta = np.concatenate([metadata(merged, preprocessing),
                               np.array([1., math.log1p(len(selected))], np.float32)])
        output.append((peaks, mask, meta))
    return output


class FPNet(nn.Module):
    def __init__(self, metadata_dim, output_bits):
        super().__init__()
        self.embedding = nn.Linear(19, 512)
        layer = nn.TransformerEncoderLayer(512, 8, 2048, .1,
                                          batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 6, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(512+metadata_dim),
                                  nn.Linear(512+metadata_dim, output_bits))

    def forward(self, peaks, mask, meta):
        encoded = self.encoder(self.embedding(peaks), src_key_padding_mask=~mask.bool())
        pooled = (encoded * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(torch.cat([pooled, meta], dim=1))


class CachedViews(Dataset):
    def __init__(self, root, training=False, seed=SEEDS[0]):
        root = Path(root)
        self.arrays = {k: np.load(root/f'{k}.npy', mmap_mode='r')
                       for k in ['peaks', 'mask', 'meta', 'owner', 'target']}
        self.layout = json.loads((root/'layout.json').read_text())
        self.training, self.seed, self.epoch = training, seed, 0
        self.selection = None

    def set_epoch(self, epoch):
        self.epoch = epoch
        rng = np.random.default_rng(self.seed+epoch)
        self.selection = []
        for group in self.layout['views']:
            # Equal single/merged training exposure, one view per identity.
            category = 'single' if rng.integers(2) == 0 else 'merged'
            eligible = group[category]
            self.selection.append(eligible[int(rng.integers(len(eligible)))])

    def __len__(self):
        return len(self.layout['identities']) if self.training else len(self.arrays['peaks'])

    def __getitem__(self, index):
        i = self.selection[index] if self.training else index
        owner = int(self.arrays['owner'][i])
        return {k: torch.from_numpy(np.array(self.arrays[k][i], copy=True))
                for k in ['peaks', 'mask', 'meta']} | {
                'target': torch.from_numpy(np.array(self.arrays['target'][owner], copy=True)),
                'index': i}


def build_cache(parquet, root, preprocessing=None, deadline=math.inf):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    if time.monotonic() > deadline:
        raise TimeoutError('Research budget exhausted before FPNet cache preparation')
    frame = pd.read_parquet(parquet)
    if preprocessing is None:
        preprocessing = fit_preprocessing(frame)
        preprocessing.update(max_peaks=MAX_PEAKS, view_metadata=['merged_flag', 'log1p_spectra_count'])
    groups = frame.groupby('identity', sort=True)
    identities = sorted(groups.groups)
    nviews = len(frame) + sum(g.ionization_mode.astype(str).nunique() for _, g in groups)
    dim = 8+sum(len(preprocessing['categories'][c])+1 for c in CATEGORIES)
    arrays = {k: np.lib.format.open_memmap(root/f'{k}.npy', mode='w+', dtype=d, shape=s)
              for k, d, s in [('peaks', np.float32, (nviews, MAX_PEAKS, 19)),
                  ('mask', bool, (nviews, MAX_PEAKS)), ('meta', np.float32, (nviews, dim)),
                  ('owner', np.int32, (nviews,)), ('target', np.uint8, (len(identities), 2048))]}
    layout, position, enum = [], 0, rdMolStandardize.TautomerEnumerator()
    targets = []
    for owner, (identity, group) in enumerate(groups):
        if time.monotonic() > deadline:
            raise TimeoutError('Research budget exhausted during FPNet cache preparation')
        records = group.sort_values('record_id').to_dict('records')
        canonical, target = canonical_target(records[0]['normalized_smiles'], identity, enum)
        arrays['target'][owner] = target
        views = {'single': [], 'merged': []}
        for category, values in [('single', [single_view(r, preprocessing) for r in records]),
                                 ('merged', merged_views(records, preprocessing))]:
            for peaks, mask, meta in values:
                for k, value in [('peaks', peaks), ('mask', mask), ('meta', meta), ('owner', owner)]:
                    arrays[k][position] = value
                views[category].append(position); position += 1
        layout.append(views)
        targets.append({'identity': identity, 'canonical_smiles': canonical})
        if (owner+1) % 5000 == 0:
            print(f'FPNet cache {Path(parquet).name}: {owner+1}/{len(identities)} identities', flush=True)
    for array in arrays.values():
        array.flush()
    assert position == nviews
    write_json(root/'layout.json', {'identities': identities, 'views': layout,
        'source_sha256': sha256(parquet), 'preprocessing': preprocessing, 'target_definition':
        'RDKit default canonical tautomer Morgan radius2 2048; identity verified'})
    pd.DataFrame(targets).to_parquet(root/'canonical_targets.parquet', index=False)
    return CachedViews(root), preprocessing


def fit_target_statistics(target):
    counts = np.asarray(target.sum(0), float)
    bits = np.flatnonzero((counts > 0) & (counts < len(target)))
    frequencies = counts[bits]/len(target)
    weights = np.minimum((len(target)-counts[bits])/counts[bits], POS_WEIGHT_CAP)
    return bits, frequencies, weights


def _canonical_pool_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        return None
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToInchiKey(mol)[:14], Chem.MolToSmiles(mol)


def query_window_mask(masses, centers):
    """Exact union of deployment windows; never consult structure identities."""
    centers = np.asarray(centers, float)
    centers = centers[np.isfinite(centers) & (centers > 0)]
    masses = np.asarray(masses, float)
    if not len(centers):
        return np.zeros(len(masses), bool)
    widths = np.maximum(centers*35e-6, .006)
    order = np.argsort(centers-widths, kind='stable')
    lower = (centers-widths)[order]
    upper = np.maximum.accumulate((centers+widths)[order])
    position = np.searchsorted(lower, masses, side='right')-1
    return np.isfinite(masses) & (position >= 0) & (masses <= upper[np.maximum(position, 0)])


def prepare_base_structure_pool(train_path, coconut_path, catalog_path, mapping, output, deadline=math.inf, query_centers=None):
    """C0 structure proposal universe; LOTUS is added only in M3/C4.

    No spectrum or held-out label is used. Preserve multiple source flags and
    validate public structures using the same official identity definition.
    """
    if time.monotonic() > deadline:
        raise TimeoutError('Research budget exhausted before structure pool preparation')
    from concurrent.futures import ProcessPoolExecutor
    from baseline import formula_mass
    metadata_frame = pd.read_parquet(train_path, columns=['normalized_smiles','molecular_formula'])
    metadata_frame = metadata_frame.drop_duplicates(['normalized_smiles','molecular_formula'])
    metadata_frame['mass'] = metadata_frame.molecular_formula.map(formula_mass)
    metadata_frame['sources'] = 'train/library'
    coconut = pd.read_parquet(coconut_path, columns=['canonical_smiles','exact_mass'])
    coconut = coconut.rename(columns={'canonical_smiles':'normalized_smiles','exact_mass':'mass'})
    coconut['sources'] = 'COCONUT'
    public = pd.read_parquet(catalog_path)
    public = public[public.formal_charge.eq(0)].copy()
    if 'provenance_json' not in public:
        raise ValueError('Public structure catalog has no preserved source provenance')
    public['sources'] = public.provenance_json.map(lambda value: ';'.join(sorted({
        str(p['source']) for p in json.loads(value)})))
    frame = pd.concat([f[['normalized_smiles','mass','sources']] for f in [metadata_frame,coconut,public]], ignore_index=True)
    frame = frame[np.isfinite(frame.mass) & frame.mass.gt(0) & frame.normalized_smiles.notna()].copy()
    before_scope = len(frame)
    if query_centers is not None:
        frame = frame[query_window_mask(frame.mass, query_centers)].copy()
    print(json.dumps({'stage':'M2_pool_mass_scope', 'source_rows':before_scope,
                     'eligible_rows':len(frame), 'query_only':query_centers is not None}), flush=True)
    smiles = sorted(set(frame.normalized_smiles)-set(mapping))
    identities = dict(mapping); canonical = {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for number,(original,result) in enumerate(zip(smiles,pool.map(_canonical_pool_smiles,smiles,chunksize=64)),1):
            if time.monotonic() > deadline:
                pool.shutdown(wait=False,cancel_futures=True)
                raise TimeoutError('Research budget exhausted during structure pool identity audit')
            if result is not None:
                identities[original],canonical[original] = result
            if number % 1000 == 0:
                print(json.dumps({'stage':'M2_pool_identity_audit', 'completed':number,
                    'total':len(smiles), 'seconds':time.monotonic()-started}), flush=True)
    frame['identity'] = frame.normalized_smiles.map(identities)
    frame = frame[frame.identity.notna()].copy()
    # Keep every alias/source row until groupwise pooling; no arbitrary mass
    # replacement across aliases is allowed.
    frame.to_parquet(output, index=False)
    write_json(Path(output).with_suffix('.manifest.json'), {
        'rows':len(frame), 'identities':int(frame.identity.nunique()),
        'source_flags':sorted(set(frame.sources)), 'canonicalization_seconds':time.monotonic()-started,
        'scope':'development deployment mass-window union' if query_centers is not None else 'full source universe',
        'source_rows_before_scope':before_scope,
        'sha256':sha256(output), 'acceptance_labels_used':False})
    return output


def grouped_logits(logits, layout):
    single = np.stack([logits[v['single']].mean(0) for v in layout['views']])
    merged = np.stack([logits[v['merged']].mean(0) for v in layout['views']])
    return single, .5*single+.5*merged


def metrics(logits, target, criterion):
    from sklearn.metrics import roc_auc_score
    logits = np.asarray(logits, np.float32); target = np.asarray(target, np.float32)
    loss = float(criterion(torch.from_numpy(logits), torch.from_numpy(target)))
    probs = 1/(1+np.exp(-np.clip(logits, -40, 40)))
    valid = (target.sum(0) > 0) & (target.sum(0) < len(target))
    per_bit = [float(roc_auc_score(target[:, i], probs[:, i])) if valid[i] else None
               for i in range(target.shape[1])]
    micro = float(roc_auc_score(target.ravel(), probs.ravel())) if np.unique(target).size == 2 else None
    macro = float(np.mean([v for v in per_bit if v is not None])) if valid.any() else None
    return {'bce': loss, 'micro_auroc': micro, 'macro_auroc': macro,
            'uncalculable_bits': int((~valid).sum()), 'per_bit_auroc': per_bit,
            'prevalence_calibration_mae': float(np.abs(probs.mean(0)-target.mean(0)).mean())}


def predict(model, dataset, bits, device, batch_size=64):
    model.eval(); output = np.empty((len(dataset), len(bits)), np.float32)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=False)
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(*(batch[k].to(device, non_blocking=True) for k in ['peaks', 'mask', 'meta']))
            value = logits.float().cpu().numpy()
            if not np.isfinite(value).all():
                raise FloatingPointError('Nonfinite development prediction')
            output[batch['index'].numpy()] = value
    return grouped_logits(output, dataset.layout)


def retry_effective_batch_fp32(model, optimizer, batches, bits, criterion, device, cpu_rng, cuda_rng=()):
    """Retry an AMP-skipped transaction; preserve examples, dropout and updates.

    GradScaler has already skipped the optimizer step and reduced its scale.
    A finite loss with FP16 gradient overflow is distinct from a corrupt loss
    or weight. Retry the full accumulation group in FP32, including earlier
    microbatches, and reject any nonfinite FP32 gradient/state.
    """
    optimizer.zero_grad(set_to_none=True)
    examples = sum(len(batch['target']) for batch in batches)
    if not examples:
        raise ValueError('Cannot retry an empty effective batch')
    total_loss = 0.
    devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(cpu_rng)
        if devices:
            torch.cuda.set_rng_state_all(cuda_rng)
        for batch in batches:
            target = batch['target'][:, bits].to(device).float()
            with torch.autocast(device.type, enabled=False):
                logits = model(*(batch[k].to(device) for k in ['peaks','mask','meta']))
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite FP32 retry loss')
            (loss*len(target)/examples).backward()
            total_loss += float(loss.detach())*len(target)
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.parameters()):
        raise FloatingPointError('Nonfinite FP32 retry gradient')
    optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise FloatingPointError('Nonfinite weight after FP32 retry')
    if any(isinstance(value,torch.Tensor) and not torch.isfinite(value).all()
           for state in optimizer.state.values() for value in state.values()):
        raise FloatingPointError('Nonfinite optimizer state after FP32 retry')
    optimizer.zero_grad(set_to_none=True)
    return total_loss


def should_finish_seed_at_boundary(history, minimum_epochs, elapsed, budget):
    """Reserve one observed epoch plus 10 seconds; preserve the 45-min cap.

    Budget exit uses time and completed rounds only, never validation quality.
    It cannot qualify a seed below the minimum completed epoch count.
    """
    if len(history)<minimum_epochs:return False
    expected=max(float(row['epoch_gpu_seconds']) for row in history[-3:])+10.
    return float(budget)-float(elapsed)<expected


def train_seed(seed, train, dev, root, bits, weights, protocol, prior_seconds=0.):
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    device = torch.device('cuda:0'); started = time.monotonic()
    raw = FPNet(train.arrays['meta'].shape[1], len(bits)).to(device)
    model = nn.DataParallel(raw) if torch.cuda.device_count() > 1 else raw
    optimizer = torch.optim.AdamW(raw.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=device))
    cpu_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32))
    scaler = torch.amp.GradScaler('cuda')
    max_epochs = protocol['fpnet']['maximum_epochs']; min_epochs = protocol['fpnet']['minimum_epochs']
    best, stale, updates, history, amp_retries = math.inf, 0, 0, [], 0
    dev_target = np.asarray(dev.arrays['target'][:, bits], np.float32)
    per_seed_budget = protocol['fpnet']['seed_seconds']
    def check_budget():
        elapsed = time.monotonic()-started
        if elapsed > per_seed_budget:
            raise TimeoutError(f'Seed {seed} exceeded frozen 45-minute budget')
        if prior_seconds+elapsed > protocol['total_research_budget_seconds']:
            raise TimeoutError('Cumulative four-hour research budget exceeded')
    stop_reason='maximum_epochs_or_patience'
    for epoch in range(1, max_epochs+1):
        if should_finish_seed_at_boundary(history,min_epochs,time.monotonic()-started,per_seed_budget):
            stop_reason='45_minute_budget_completed_epoch_boundary'
            print(json.dumps({'stage':'M2_seed_budget_stop','seed':seed,
                'completed_epochs':len(history),'selected_bce':best,
                'actual_optimizer_updates':updates,'seconds':time.monotonic()-started}),flush=True)
            break
        check_budget(); train.seed = seed; train.set_epoch(epoch)
        # Two cards: batch64 split32/card; one card: batch32 and accumulate2.
        batch_size = 64 if torch.cuda.device_count() > 1 else 32
        accumulation = 64//batch_size
        loader = DataLoader(train, batch_size=batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed+epoch), num_workers=2,
            pin_memory=True, drop_last=False, persistent_workers=False)
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_start = time.monotonic(); epoch_loss, seen = 0., 0
        pending_batches, group_original_loss = [], 0.
        for card in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(card)
        for step, batch in enumerate(loader):
            check_budget()
            if not pending_batches:
                group_cpu_rng = torch.get_rng_state()
                group_cuda_rng = torch.cuda.get_rng_state_all()
            pending_batches.append(batch)
            fraction = (epoch-1)+(step+1)/len(loader)
            rate = fraction/2 if fraction < 2 else .5*(1+math.cos(math.pi*(fraction-2)/(max_epochs-2)))
            for group in optimizer.param_groups:
                group['lr'] = 1e-4*rate
            target = batch['target'][:, bits].to(device, non_blocking=True).float()
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(*(batch[k].to(device, non_blocking=True) for k in ['peaks', 'mask', 'meta']))
                loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            # Correct scaling of the final short accumulation group.
            group_size = min(accumulation, len(loader)-(step//accumulation)*accumulation)
            scaler.scale(loss/group_size).backward()
            epoch_loss += float(loss.detach())*len(target); seen += len(target)
            group_original_loss += float(loss.detach())*len(target)
            if (step+1) % accumulation == 0 or step+1 == len(loader):
                old_scale = scaler.get_scale()
                scaler.step(optimizer); scaler.update()
                if scaler.get_scale() < old_scale:
                    replacement_loss = retry_effective_batch_fp32(model, optimizer,
                        pending_batches, bits, criterion, device, group_cpu_rng, group_cuda_rng)
                    epoch_loss += replacement_loss-group_original_loss
                    amp_retries += 1
                    print(json.dumps({'stage':'M2_AMP_FP32_retry','seed':seed,'epoch':epoch,
                        'step':step+1,'amp_skipped_steps':amp_retries,
                        'effective_examples':sum(len(b['target']) for b in pending_batches),
                        'optimizer_update_succeeded':True}),flush=True)
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                pending_batches, group_original_loss = [], 0.
            if (step+1) % 100 == 0:
                print(json.dumps({'stage': 'M2_training', 'seed': seed, 'epoch': epoch,
                    'step': step+1, 'steps': len(loader), 'seconds': time.monotonic()-started,
                    'optimizer_updates': updates}), flush=True)
        single, fused = predict(model, dev, bits, device)
        check_budget(); result = metrics(fused, dev_target, cpu_loss)
        peak = [int(torch.cuda.max_memory_allocated(card)) for card in range(torch.cuda.device_count())]
        if any(v > 14*(1 << 30) for v in peak):
            raise MemoryError('FPNet exceeded frozen target 14 GiB/card')
        row = {'epoch': epoch, 'train_bce': epoch_loss/seen, 'development': result,
               'learning_rate': optimizer.param_groups[0]['lr'], 'physical_batch': batch_size,
               'effective_batch': 64, 'accumulation': accumulation, 'identity_examples': seen,
               'epoch_gpu_seconds': time.monotonic()-epoch_start,
               'elapsed_seconds': time.monotonic()-started, 'peak_vram_bytes': peak,
               'optimizer_updates': updates, 'amp_fp32_retries':amp_retries}
        history.append(row); write_json(root/f'seed_{seed}_history.json', history)
        torch.save({'state_dict':raw.state_dict(), 'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(), 'completed_epoch':epoch, 'seed':seed,
            'bits':bits.tolist(), 'optimizer_updates':updates, 'amp_fp32_retries':amp_retries,
            'torch_cpu_rng_state':torch.get_rng_state(),
            'torch_cuda_rng_states':torch.cuda.get_rng_state_all()}, root/f'seed_{seed}_last_complete.pt')
        print(json.dumps({k:v for k,v in row.items() if k != 'development'} | {
            'development_bce': result['bce'], 'seed': seed}), flush=True)
        if result['bce'] < best:
            best, stale = result['bce'], 0
            torch.save({'state_dict': raw.state_dict(), 'selected_epoch': epoch,
                'seed': seed, 'optimizer_updates': updates, 'bits': bits.tolist(),
                'metadata_dim': train.arrays['meta'].shape[1], 'preprocessing': train.layout['preprocessing'],
                'architecture': {'layers':6, 'hidden':512, 'heads':8, 'dropout':.1},
                'development_bce': best}, root/f'fpnet_seed_{seed}.pt')
            np.save(root/f'seed_{seed}_single_logits.npy', single)
            np.save(root/f'seed_{seed}_fused_logits.npy', fused)
        else:
            stale += 1
        if epoch >= min_epochs and stale >= 4:
            break
    if len(history) < min_epochs:
        raise ValueError('Seed did not complete the minimum six epochs')
    return {'seed':seed, 'completed_epochs':len(history), 'selected_bce':best,
            'actual_optimizer_updates':updates, 'seconds':time.monotonic()-started, 'stop_reason':stop_reason,
            'checkpoint_sha256':sha256(root/f'fpnet_seed_{seed}.pt')}


def development_candidate_gate(frame, pool, bits, ensemble, single, root, deadline):
    """Compare FPNet and random over the identical deployment mass windows.

    Every structure is scored without knowing the true identity. Presence of
    truth is used only afterwards by this evaluator to define the conditional
    subgroup specified in FR-11. The acceptance file is never read.
    """
    from baseline import ADDUCT_MASS
    from casmi_ml.v1_routing import paired
    pool = pool.sort_values(['mass', 'identity', 'normalized_smiles'], kind='stable').reset_index(drop=True)
    masses = pool.mass.to_numpy(float)
    cache, enum, cases = {}, rdMolStandardize.TautomerEnumerator(), []
    random = np.random.default_rng(SEEDS[0])
    for n, (truth, group) in enumerate(frame.groupby('identity', sort=True)):
        if time.monotonic() > deadline:
            raise TimeoutError('Cumulative research budget exceeded in FPNet development gate')
        center = float(np.median(group.precursor_mz-group.adduct.map(ADDUCT_MASS)))
        window = max(center*35e-6, .006)
        start = np.searchsorted(masses, center-window)
        stop = np.searchsorted(masses, center+window, side='right')
        candidates = pool.iloc[start:stop].drop_duplicates('identity', keep='first')
        identities = candidates.identity.tolist()
        vectors = []
        for row in candidates.itertuples():
            if row.identity not in cache:
                canonical, fp = canonical_target(row.normalized_smiles, row.identity, enum)
                cache[row.identity] = (canonical, fp[bits])
            vectors.append(cache[row.identity][1])
        if len(vectors):
            matrix = np.asarray(vectors, np.float32)
            fused_order = np.lexsort((np.array(identities), -(matrix @ ensemble[n])))
            single_order = np.lexsort((np.array(identities), -(matrix @ single[n])))
            random_order = random.permutation(len(identities))
        else:
            fused_order = single_order = random_order = np.array([], int)
        # Truth begins here, after every candidate score/order is computed.
        def rr(order):
            ids = [identities[i] for i in order[:25]]
            return 1/(ids.index(truth)+1) if truth in ids else 0.
        cases.append({'identity':truth, 'mass_window_candidates':len(identities),
            'truth_in_mass_window':truth in identities, 'fused_rr':rr(fused_order),
            'single_rr':rr(single_order), 'random_rr':rr(random_order),
            'candidate_identities':identities, 'fused_order':fused_order.tolist(),
            'single_order':single_order.tolist(), 'random_order':random_order.tolist()})
        if (n+1) % 25 == 0:
            print(json.dumps({'stage':'M2_conditional_candidate_gate', 'completed':n+1,
                'total':int(frame.identity.nunique()), 'cached_fingerprints':len(cache)}), flush=True)
    conditional = [c for c in cases if c['truth_in_mass_window']]
    if not conditional:
        raise ValueError('No development truths in the deployment mass windows')
    difference = paired([c['fused_rr']-c['random_rr'] for c in conditional], seed=SEEDS[0], repeats=10000)
    blend_delta = float(np.mean([c['fused_rr']-c['single_rr'] for c in conditional]))
    selected_view = 'single' if blend_delta < -.001 else 'fused'
    selected_comparison = paired([c[selected_view+'_rr']-c['random_rr'] for c in conditional],
                                 seed=SEEDS[0], repeats=10000)
    report = {'queries':len(cases), 'conditional_truth_present':len(conditional),
              'conditional_fused_vs_random':difference, 'fused_minus_single_mrr':blend_delta,
              'selected_view':selected_view, 'selected_vs_random':selected_comparison,
              'passed':selected_comparison['ci95'][0] > 0, 'acceptance_opened':False,
              'mass_window':'unchanged 35 ppm with .006 Da floor; no nearest fallback for structure pool'}
    write_json(root/'M2_development_candidate_cases.json', cases)
    write_json(root/'M2_development_candidate_gate.json', report)
    return report


def verified_cached_views(cache_root, parquet, training=False):
    """Reuse frozen views without rebuilding them; verify provenance and layout."""
    cache_root, parquet = Path(cache_root), Path(parquet)
    cached = CachedViews(cache_root, training=training)
    if cached.layout['source_sha256'] != sha256(parquet):
        raise ValueError('Cached views do not match the frozen source parquet')
    identities = cached.layout['identities']
    views = cached.layout['views']
    if not identities or identities != sorted(set(identities)) or len(views) != len(identities):
        raise ValueError('Invalid cached identity ordering')
    n = len(cached.arrays['owner'])
    if cached.arrays['target'].shape != (len(identities), 2048):
        raise ValueError('Invalid cached target dimensions')
    if cached.arrays['peaks'].shape != (n, MAX_PEAKS, 19) or cached.arrays['mask'].shape != (n, MAX_PEAKS):
        raise ValueError('Invalid cached peak dimensions')
    if len(cached.arrays['meta']) != n:
        raise ValueError('Invalid cached metadata dimensions')
    seen = np.zeros(n, bool)
    for owner, group in enumerate(views):
        for category in ('single', 'merged'):
            ids = np.asarray(group[category], dtype=np.int64)
            if not len(ids) or np.any(ids < 0) or np.any(ids >= n):
                raise ValueError('Invalid cached view index')
            if len(np.unique(ids)) != len(ids) or seen[ids].any() or np.any(cached.arrays['owner'][ids] != owner):
                raise ValueError('Cached view ownership or uniqueness mismatch')
            seen[ids] = True
    if not seen.all():
        raise ValueError('Unassigned cached views')
    for start in range(0, n, 8192):
        for key in ('peaks', 'meta'):
            if not np.isfinite(cached.arrays[key][start:start+8192]).all():
                raise ValueError('Nonfinite cached features')
        if not cached.arrays['mask'][start:start+8192].any(1).all():
            raise ValueError('Empty cached mask')
    if not np.isin(cached.arrays['target'], [0, 1]).all():
        raise ValueError('Nonbinary cached fingerprints')
    audit = {'source_sha256':cached.layout['source_sha256'], 'identities':len(identities), 'views':n,
        'files':{name:sha256(cache_root/name) for name in
                 ['layout.json','peaks.npy','mask.npy','meta.npy','owner.npy','target.npy','canonical_targets.parquet']}}
    return cached, audit


def cuda_amp_recovery_preflight(train, bits, weights, root):
    """Bounded real-architecture GPU transaction check before fresh seeds."""
    import copy
    started = time.monotonic()
    device = torch.device('cuda:0')
    torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    train.set_epoch(0)
    batches = list(DataLoader(torch.utils.data.Subset(train, range(min(64, len(train)))), batch_size=32))
    raw = FPNet(train.arrays['meta'].shape[1], len(bits)).to(device)
    direct_raw = copy.deepcopy(raw)
    wrap = lambda m: nn.DataParallel(m) if torch.cuda.device_count() > 1 else m
    model, direct = wrap(raw), wrap(direct_raw)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=1e-4, weight_decay=1e-4)
    direct_optimizer = torch.optim.AdamW(direct_raw.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32, device=device))
    scaler = torch.amp.GradScaler('cuda')
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    for batch in batches:
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(*(batch[k].to(device) for k in ['peaks','mask','meta']))
            loss = criterion(logits, batch['target'][:,bits].to(device).float())
        if not torch.isfinite(loss):raise FloatingPointError('GPU preflight AMP loss nonfinite')
        scaler.scale(loss/len(batches)).backward()
    # Deliberately exercise GradScaler's protection, never a training example skip.
    next(p for p in raw.parameters() if p.grad is not None).grad.view(-1)[0] = math.inf
    previous_scale = scaler.get_scale()
    scaler.step(optimizer); scaler.update()
    skipped = scaler.get_scale() < previous_scale and not optimizer.state
    unchanged = all(torch.equal(a,b) for a,b in zip(raw.parameters(),direct_raw.parameters()))
    if not skipped or not unchanged:raise ValueError('GPU preflight AMP skip did not preserve weights')
    retry_effective_batch_fp32(model, optimizer, batches, bits, criterion, device, cpu_rng, cuda_rng)
    retry_effective_batch_fp32(direct, direct_optimizer, batches, bits, criterion, device, cpu_rng, cuda_rng)
    difference = max(float((a-b).detach().abs().max()) for a,b in zip(raw.parameters(),direct_raw.parameters()))
    if not all(torch.allclose(a,b,rtol=1e-4,atol=1e-5) for a,b in zip(raw.parameters(),direct_raw.parameters())):
        raise ValueError('GPU preflight FP32 replay parity failed')
    if not all(float(s['step']) == 1 for s in optimizer.state.values()):
        raise ValueError('GPU preflight optimizer did not update exactly once')
    report = {'passed':True,'torch_version':torch.__version__,'devices':[torch.cuda.get_device_name(i)
        for i in range(torch.cuda.device_count())], 'real_architecture':True,'effective_examples':sum(len(b['target']) for b in batches),
        'forced_overflow_skipped':skipped,'skipped_weights_unchanged':unchanged,
        'successful_recovery_updates':1,'fp32_replay_max_absolute_difference':difference,
        'seconds':time.monotonic()-started,'preflight_weights_used_for_training':False}
    write_json(Path(root)/'gpu_amp_preflight.json',report)
    print(json.dumps({'stage':'M2_GPU_AMP_preflight',**report}),flush=True)
    del model, direct, raw, direct_raw, optimizer, direct_optimizer
    torch.cuda.empty_cache()
    return report


def run(foundation, root, protocol, candidate_pool=None, preparation_seconds=0.,
        cache_root=None, experiment_prior_seconds=None, historical_worker_seconds=0., completed_seed_root=None, completed_seed_declaration=None):
    foundation, root = Path(foundation), Path(root)
    root.mkdir(parents=True, exist_ok=True); started = time.monotonic()
    runtime = json.loads((foundation/'runtime_profile.json').read_text())
    prior_seconds = float(runtime[-1]['elapsed_seconds'])+float(preparation_seconds)
    if experiment_prior_seconds is not None:
        # Explicit new user-authorized experiment; preserve lifetime costs separately.
        prior_seconds = float(experiment_prior_seconds)+float(preparation_seconds)
    state = json.loads((foundation/'run_status.json').read_text())
    if state['stage'] != 'M1_foundation_complete':
        raise ValueError('Foundation did not finish; M2 forbidden')
    if rdBase.rdkitVersion != protocol['rdkit_version']:
        raise ValueError('M2 target RDKit version mismatch')
    status = {'stage':'M2_preparing', 'acceptance_opened':False, 'competition_submission_allowed':False,
              'historical_worker_seconds':float(historical_worker_seconds), 'training_restart':'all seeds from scratch' if completed_seed_root is None else 'recover current experiment completed seed; remaining seeds fresh'}
    write_json(root/'run_status.json', status)
    try:
        if prior_seconds >= protocol['total_research_budget_seconds']:
            raise TimeoutError('Cumulative research budget already exhausted before FPNet')
        if not torch.cuda.is_available():
            raise RuntimeError('M2 requires GPU; no CPU training fallback')
        deadline = started+protocol['total_research_budget_seconds']-prior_seconds
        if cache_root is None:
            train_cache, preprocessing = build_cache(foundation/'train.parquet', root/'train_cache', deadline=deadline)
            dev, _ = build_cache(foundation/'development.parquet', root/'development_cache', preprocessing, deadline=deadline)
            train = CachedViews(root/'train_cache', training=True)
        else:
            train, train_audit = verified_cached_views(Path(cache_root)/'train_cache', foundation/'train.parquet', True)
            dev, dev_audit = verified_cached_views(Path(cache_root)/'development_cache', foundation/'development.parquet')
            preprocessing = train.layout['preprocessing']
            if preprocessing != dev.layout['preprocessing']:
                raise ValueError('Cached train/dev preprocessing mismatch')
            if set(train.layout['identities']) & set(dev.layout['identities']):
                raise ValueError('Cached train/dev identity leakage')
            write_json(root/'cache_reuse_audit.json', {'train':train_audit, 'development':dev_audit,
                'rebuilt':False, 'weights_reused':False, 'acceptance_file_read':False})
            print(json.dumps({'stage':'M2_verified_cache_reuse','train':len(train),
                'development':len(dev.layout['identities']), 'seconds':time.monotonic()-started}),flush=True)
        bits, frequencies, weights = fit_target_statistics(train.arrays['target'])
        if cache_root is not None:
            cuda_amp_recovery_preflight(train, bits, weights, root)
        write_json(root/'train_fitted_statistics.json', {'bits':bits.tolist(),
            'frequencies':frequencies.tolist(), 'pos_weights':weights.tolist(),
            'pos_weight_cap':POS_WEIGHT_CAP, 'identity_weighting':'uniform; one view/identity/epoch',
            'merged_training_probability':.5, 'train_identity_count':len(train),
            'preparation_seconds':time.monotonic()-started, 'preprocessing':preprocessing,
            'acceptance_opened':False})
        results = []
        for seed in SEEDS:
            prior = prior_seconds+time.monotonic()-started
            if seed==SEEDS[0] and completed_seed_root is not None:
                from casmi_ml.np_pairtail_seed_recovery import recover_completed_seed
                results.append(recover_completed_seed(completed_seed_root,root,train,dev,bits,weights,protocol,completed_seed_declaration))
            else:
                results.append(train_seed(seed, train, dev, root, bits, weights, protocol, prior))
        ensemble = np.mean([np.load(root/f'seed_{s}_fused_logits.npy') for s in SEEDS], axis=0)
        single = np.mean([np.load(root/f'seed_{s}_single_logits.npy') for s in SEEDS], axis=0)
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, dtype=torch.float32))
        target = np.asarray(dev.arrays['target'][:, bits], np.float32)
        constant = np.broadcast_to(np.log(frequencies/(1-frequencies)), target.shape).copy().astype(np.float32)
        const_metrics, ens_metrics = metrics(constant, target, loss), metrics(ensemble, target, loss)
        np.save(root/'ensemble_fused_logits.npy', ensemble); np.save(root/'ensemble_single_logits.npy', single)
        write_json(root/'development_identity_order.json', dev.layout['identities'])
        bce_pass = ens_metrics['bce'] <= .9*const_metrics['bce']
        write_json(root/'M2_training_summary.json', {'seeds':results, 'ensemble':ens_metrics,
            'constant_train_frequency':const_metrics, 'bce_gate_pass':bce_pass,
            'candidate_ranking_gate':'pending development same-window conditional-truth evaluation',
            'acceptance_opened':False})
        if not bce_pass:
            raise ValueError('M2 ensemble BCE did not beat train-frequency constant by 10%')
        if candidate_pool is None:
            raise ValueError('Candidate mass-pool asset is missing; M2 ranking gate cannot be skipped')
        gate = development_candidate_gate(pd.read_parquet(foundation/'development.parquet'),
            pd.read_parquet(candidate_pool), bits, ensemble, single, root,
            started+protocol['total_research_budget_seconds']-prior_seconds)
        if not gate['passed']:
            raise ValueError('FPNet development same-window ordering did not significantly beat random')
        write_json(root/'run_status.json', status | {'stage':'M2_complete',
            'elapsed_seconds':time.monotonic()-started, 'prior_seconds':prior_seconds,
            'cumulative_seconds':prior_seconds+time.monotonic()-started})
    except Exception as exc:
        write_json(root/'run_status.json', status | {'stage':'M2_failed', 'error_type':type(exc).__name__,
            'error':str(exc), 'elapsed_seconds':time.monotonic()-started,
            'cumulative_seconds':prior_seconds+time.monotonic()-started})
        raise
