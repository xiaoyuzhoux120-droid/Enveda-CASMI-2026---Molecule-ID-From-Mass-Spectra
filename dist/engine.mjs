// Browser adaptation of baseline.py's spectral vectorization and cosine ranking.
export const ADDUCTS = {'[M+H]+': 1.007276466621, '[M+Na]+': 22.989218, '[M+NH4]+': 18.033825553};
const rint = x => { const n = Math.floor(x); return x - n === 0.5 ? n + n % 2 : Math.round(x); };
export const binId = mz => rint(mz * 10);
export function parsePeaks(text) {
  const peaks = [];
  for (const [index, raw] of text.split(/\r?\n/).entries()) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    if (/^(m\/?z|mass)[\s,;\t]/i.test(line)) continue;
    const values = line.split(/[\s,;]+/);
    if (values.length !== 2) throw Error(`Line ${index + 1}: enter two columns, m/z and intensity.`);
    const [mz, intensity] = values.map(Number);
    if (!Number.isFinite(mz) || mz <= 0 || !Number.isFinite(intensity) || intensity < 0)
      throw Error(`Line ${index + 1}: m/z must be positive and intensity must be zero or positive.`);
    peaks.push([mz, intensity]);
  }
  if (!peaks.length) throw Error('Add at least one peak, or choose an example.');
  if (peaks.length > 10000) throw Error('Use a spectrum with at most 10,000 peaks.');
  const max = Math.max(...peaks.map(p => p[1]));
  if (max <= 0) throw Error('At least one intensity must be greater than zero.');
  return peaks.map(([m,i]) => [m,i/max]).sort((a,b) => a[0]-b[0]);
}
export function vectorize(peaks) {
  const usable = peaks.filter(([m,i]) => Number.isFinite(m) && Number.isFinite(i) && i >= .01 && m >= 40 && m < 1250)
    .sort((a,b) => b[1]-a[1]).slice(0,100);
  const bins = new Map();
  for (const [mz,i] of usable) {
    const bin = binId(mz), weight = Math.sqrt(i) * Math.sqrt(mz/100);
    bins.set(bin, Math.max(bins.get(bin) || 0, weight));
  }
  const norm = Math.sqrt([...bins.values()].reduce((s,v) => s+v*v, 0));
  return new Map([...bins].map(([bin,v]) => [bin,v/norm]));
}
export function cosine(a,b) {
  let score=0, matched=0;
  for (const [bin,weight] of a) if (b.has(bin)) { score += weight*b.get(bin); matched++; }
  return {score:Math.min(1,Math.max(0,score)),matched};
}
export function search(library, peaks, {precursor,adduct='[M+H]+',ppm=35,massFilter=true}={}) {
  if (!Number.isFinite(precursor) || precursor <= 0) throw Error('Enter a positive precursor m/z.');
  if (!(adduct in ADDUCTS)) throw Error('Choose a supported positive, singly charged adduct.');
  if (!Number.isFinite(ppm) || ppm <= 0 || ppm > 1000) throw Error('Mass tolerance must be between 0 and 1,000 ppm.');
  const vector = vectorize(peaks);
  if (!vector.size) throw Error('No usable peaks: use m/z 40–1,250 with at least 1% relative intensity.');
  const neutral = precursor-ADDUCTS[adduct];
  if (neutral <= 0) throw Error('Precursor m/z must exceed the adduct mass.');
  const delta=Math.max(neutral*ppm*1e-6,.006);
  const ranked=[];
  for (const compound of library) {
    if (massFilter && Math.abs(compound.mass-neutral)>delta) continue;
    let best=null;
    for (const reference of compound.spectra) {
      const result=cosine(vector,vectorize(reference.peaks));
      if (!best || result.score>best.score) best={...result,reference};
    }
    ranked.push({...compound,...best,ppmError:(compound.mass-neutral)/neutral*1e6});
  }
  ranked.sort((a,b)=>b.score-a.score || Math.abs(a.ppmError)-Math.abs(b.ppmError));
  return {ranked,neutral,delta,usedPeaks:vector.size};
}
// Exploratory mass differences only. A match is not a functional-group assignment.
export function neutralLosses(peaks,precursor) {
  const losses=[['H₂O',18.010565],['CO',27.994915],['CO₂',43.989829],['C₆H₁₀O₅',162.052823]];
  return losses.map(([formula,mass])=>{
    const candidates=peaks.filter(([mz,i])=>i>=.01 && mz<precursor &&
      Math.abs(precursor-mz-mass)<=Math.max(precursor*10e-6,.002)+Math.max(mz*10e-6,.002));
    candidates.sort((a,b)=>b[1]-a[1]);
    return {formula,mass,peak:candidates[0]};
  }).filter(x=>x.peak);
}
