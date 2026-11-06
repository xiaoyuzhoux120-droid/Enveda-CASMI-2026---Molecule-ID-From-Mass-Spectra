import {parsePeaks, search, vectorize, neutralLosses, binId} from './engine.mjs';
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const BRANCH='codex/spectra-pages';
const REPO='https://github.com/xiaoyuzhoux120-droid/Enveda-CASMI-2026---Molecule-ID-From-Mass-Spectra';
document.querySelectorAll('.repo-link').forEach(a=>a.href=`${REPO}/tree/${BRANCH}`);
let data,activeExample=null,result=null,selection=0,currentPeaks=[],stale=false;
function setError(message='') { $('error').textContent=message; $('error').hidden=!message; }
function peakText(peaks) { return peaks.map(([m,i])=>`${m.toFixed(4)}, ${(i*100).toFixed(3)}`).join('\n'); }
function markChanged() {
  stale=true; $('export').disabled=true;
  $('search-status').textContent='Inputs changed — search again to update';
  $('search-status').classList.add('stale-note');
  $('peak-count').textContent=`${$('peaks').value.split(/\n/).filter(x=>x.trim()&&!x.startsWith('#')).length} lines`;
}
function loadExample(id) {
  const example=data.examples.find(e=>e.compoundId===id);
  activeExample=example;
  $('precursor').value=example.precursor;
  $('adduct').value=example.adduct;
  $('peaks').value=peakText(example.peaks);
  $('query-origin').innerHTML=`Separate query · ${esc(example.energy)} · <a target="_blank" rel="noopener" href="${esc(example.source)}">${esc(example.id)}</a>`;
  document.querySelectorAll('.example-button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.id===id)));
  runSearch();
}
function runSearch() {
  try {
    const peaks=parsePeaks($('peaks').value);
    const precursor=Number($('precursor').value),ppm=Number($('ppm').value);
    const outcome=search(data.library,peaks,{precursor,adduct:$('adduct').value,ppm,massFilter:$('mass-filter').checked});
    result=outcome; currentPeaks=peaks; selection=0; stale=false; setError();
    $('export').disabled=!result.ranked.length;
    $('peak-count').textContent=`${peaks.length} peaks`;
    $('search-status').classList.remove('stale-note');
    $('search-status').textContent=`${outcome.usedPeaks} bins compared · ${outcome.ranked.length} of ${data.library.length} compounds retained`;
    renderCandidates(); renderSelected(); renderLosses(peaks,precursor);
  } catch(error) { setError(error.message); markChanged(); }
}
function renderCandidates() {
  $('candidate-count').textContent=`${result.ranked.length} candidate${result.ranked.length===1?'':'s'} · neutral mass ${result.neutral.toFixed(4)} Da`;
  if(!result.ranked.length) {
    $('candidates').innerHTML='<div class="empty-state"><strong>No reference compound within this mass window.</strong>Try a different example, check precursor and adduct, or turn off the mass filter to compare all four compounds. This small library cannot identify an absent compound.</div>';
    return;
  }
  $('candidates').innerHTML=result.ranked.map((c,i)=>`<button class="candidate-row ${i===selection?'selected':''}" type="button" aria-pressed="${i===selection}" data-index="${i}" aria-label="Compare ${esc(c.name)}, cosine similarity ${c.score.toFixed(3)}"><span class="rank-number">${String(i+1).padStart(2,'0')}</span><span class="candidate-name"><strong>${esc(c.name)}</strong><span>${esc(c.formula)} · ${c.mass.toFixed(4)} Da</span></span><span class="score-bar" aria-hidden="true"><span style="width:${c.score*100}%"></span></span><span class="score-value">${c.score.toFixed(3)}</span></button>`).join('');
  $('candidates').querySelectorAll('button').forEach(button=>button.addEventListener('click',()=>{selection=Number(button.dataset.index);renderCandidates();renderSelected();}));
}
function renderSelected() {
  const candidate=result.ranked[selection];
  if(!candidate) { $('candidate-detail').innerHTML=''; $('chart-subtitle').textContent='Query spectrum · no matching reference'; drawChart(currentPeaks,[]); return; }
  $('chart-subtitle').textContent=`Query vs. ${candidate.name} · reference ${candidate.reference.energy}`;
  $('candidate-detail').innerHTML=`<div class="candidate-detail"><img class="molecule-preview" src="${esc(candidate.image)}" alt="Molecular structure of ${esc(candidate.name)}"><div><p class="detail-eyebrow">SELECTED CANDIDATE</p><p class="detail-title">${esc(candidate.name)}</p><div class="detail-data"><div><span>Mass error</span><strong>${candidate.ppmError>=0?'+':''}${candidate.ppmError.toFixed(2)} ppm</strong></div><div><span>Shared peak bins</span><strong>${candidate.matched} / ${result.usedPeaks}</strong></div></div><a class="record-link" href="${esc(candidate.reference.source)}" target="_blank" rel="noopener">Inspect best reference ↗</a></div></div>`;
  drawChart(currentPeaks,candidate.reference.peaks);
}
function drawChart(query,reference) {
  const width=800,height=250,left=40,right=18,top=15,baseline=112,amplitude=80;
  const maxMz=Math.max(100,...query.map(p=>p[0]),...reference.map(p=>p[0]));
  const step=maxMz>400?100:50;
  const axisMax=Math.ceil((maxMz+8)/step)*step;
  const x=m=>left+m/axisMax*(width-left-right);
  const qBins=vectorize(query),rBins=vectorize(reference);
  let svg='<title>Query spectrum mirrored against selected reference</title><desc>Green query peaks point up; terracotta reference peaks point down. Intensities are normalized separately. Labels show the strongest query peaks.</desc>';
  for(let mz=0;mz<=axisMax;mz+=step) svg+=`<line x1="${x(mz)}" y1="${top}" x2="${x(mz)}" y2="207" stroke="#edf0e6"/><text x="${x(mz)}" y="225" text-anchor="middle">${mz}</text>`;
  svg+=`<line x1="${left}" y1="${baseline}" x2="${width-right}" y2="${baseline}" stroke="#d4ddcb"/><text x="27" y="37" text-anchor="end">100</text><text x="27" y="116" text-anchor="end">0</text><text x="27" y="196" text-anchor="end">100</text><text x="${width-right}" y="243" text-anchor="end">m/z</text><text x="${left}" y="243">Relative intensity (%)</text>`;
  for(const [mz,i] of reference) if(i>=.01) svg+=`<line x1="${x(mz)}" y1="${baseline}" x2="${x(mz)}" y2="${baseline+i*amplitude}" stroke="${qBins.has(binId(mz))?'#bd8057':'#d7b79d'}" stroke-width="1.7"><title>Reference: ${mz.toFixed(4)} m/z, ${(i*100).toFixed(1)}%</title></line>`;
  for(const [mz,i] of query) if(i>=.01) svg+=`<line x1="${x(mz)}" y1="${baseline}" x2="${x(mz)}" y2="${baseline-i*amplitude}" stroke="${rBins.has(binId(mz))?'#397256':'#88aa8e'}" stroke-width="1.8"><title>Query: ${mz.toFixed(4)} m/z, ${(i*100).toFixed(1)}%</title></line>`;
  const labeled=[];
  for(const [mz,i] of [...query].sort((a,b)=>b[1]-a[1])) {
    if(labeled.length>=4) break;
    if(labeled.some(m=>Math.abs(x(m)-x(mz))<50)) continue;
    labeled.push(mz);svg+=`<text x="${x(mz)}" y="${baseline-i*amplitude-7}" text-anchor="middle" style="fill:#5b8061">${mz.toFixed(2)}</text>`;
  }
  $('spectrum-chart').innerHTML=svg;
}
function renderLosses(peaks,precursor) {
  const matches=neutralLosses(peaks,precursor);
  $('loss-summary').textContent=matches.length?`${matches.length} possible neutral-loss match${matches.length===1?'':'es'} within conservative 10 ppm / 0.002 Da measurement bounds.`:'No H₂O, CO, CO₂, or C₆H₁₀O₅ precursor-loss matches at the current measurement tolerance.';
  $('loss-tags').innerHTML=matches.map(m=>`<span class="loss-tag">${esc(m.formula)} · Δ${m.mass.toFixed(4)} Da</span>`).join('');
}
function exportCsv() {
  if(stale||!result?.ranked.length)return;
  const quote=x=>'"'+String(x).replaceAll('"','""')+'"';
  const rows=[['rank','compound','formula','cosine_similarity','mass_error_ppm','shared_bins','reference_accession','reference_source'],...result.ranked.map((c,i)=>[i+1,c.name,c.formula,c.score,c.ppmError,c.matched,c.reference.id,c.reference.source])];
  const blob=new Blob([rows.map(row=>row.map(quote).join(',')).join('\n')],{type:'text/csv;charset=utf-8'});
  const link=document.createElement('a');const url=URL.createObjectURL(blob);link.href=url;link.download='spectra-candidate-ranking.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
try {
  const response=await fetch('data/library.json');
  if(!response.ok) throw Error('The library could not be loaded. Please refresh this page.');
  data=await response.json();
  $('examples').innerHTML=data.library.map(c=>`<button type="button" class="example-button" aria-pressed="false" data-id="${esc(c.id)}"><span class="example-dot" aria-hidden="true"></span>${esc(c.name)}</button>`).join('');
  $('examples').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>loadExample(b.dataset.id)));
  $('search').addEventListener('click',runSearch);
  $('reset').addEventListener('click',()=>{$('mass-filter').checked=true;$('ppm').value=35;loadExample('genistein');});
  $('export').addEventListener('click',exportCsv);
  ['precursor','adduct','peaks','mass-filter','ppm'].forEach(id=>$(id).addEventListener('input',()=>{
    markChanged();
    if(['peaks','precursor','adduct'].includes(id)) {
      activeExample=null;
      $('query-origin').textContent='Custom spectrum · processed only in your browser';
      document.querySelectorAll('.example-button').forEach(b=>b.setAttribute('aria-pressed','false'));
    }
  }));
  $('import').addEventListener('click',()=>$('file-input').click());
  $('file-input').addEventListener('change',async()=>{
    const file=$('file-input').files[0];if(!file)return;
    try {
      if(file.size>1024*1024)throw Error('Choose a peak file smaller than 1 MB.');
      const text=await file.text();parsePeaks(text);$('peaks').value=text;activeExample=null;
      $('query-origin').textContent=`Imported ${file.name} · processed only in your browser`;
      document.querySelectorAll('.example-button').forEach(b=>b.setAttribute('aria-pressed','false'));
      markChanged();setError();
    }catch(error){setError(error.message);}
    $('file-input').value='';
  });
  $('loading').hidden=true;$('search-ui').hidden=false;loadExample('genistein');
}catch(error){$('loading').textContent=error.message;$('loading').setAttribute('role','alert');}
