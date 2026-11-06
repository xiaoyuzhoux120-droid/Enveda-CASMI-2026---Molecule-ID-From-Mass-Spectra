import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {parsePeaks,vectorize,cosine,search,neutralLosses,ADDUCTS} from '../dist/engine.mjs';
const data=JSON.parse(fs.readFileSync(new URL('../dist/data/library.json',import.meta.url)));
const oracle=JSON.parse(fs.readFileSync(new URL('./python-oracle.json',import.meta.url)));
test('browser scores agree with unchanged Python baseline on 16 cross-spectrum comparisons',()=>{
  for(const example of data.examples) {
    const result=search(data.library,example.peaks,{precursor:example.precursor,massFilter:false});
    for(const row of result.ranked) {
      const expected=oracle.scores.find(s=>s.query===example.id&&s.compound===row.id);
      assert.ok(Math.abs(row.score-expected.score)<1e-12,`${example.name} / ${row.name}`);
    }
  }
});
test('half-bin rounding agrees with numpy, including ties to even',()=>{
  const actual=vectorize(oracle.roundingPeaks);
  assert.deepEqual([...actual.keys()].sort(),oracle.roundingVector.map(v=>v[0]).sort());
  for(const [k,v] of oracle.roundingVector)assert.ok(Math.abs(actual.get(k)-v)<1e-12);
});
test('mass filtering retains the correct mass for every separate query',()=>{
  for(const example of data.examples){const r=search(data.library,example.peaks,{precursor:example.precursor});assert.equal(r.ranked.length,1);assert.equal(r.ranked[0].id,example.compoundId);}
});
test('mass window eliminates a spectrally similar but incompatible candidate',()=>{
  const e=data.examples.find(x=>x.name==='Kaempferol');
  assert.equal(search(data.library,e.peaks,{precursor:e.precursor,massFilter:false}).ranked[0].name,'Genistein');
  assert.equal(search(data.library,e.peaks,{precursor:e.precursor}).ranked[0].name,'Kaempferol');
});
test('no hidden fallback when the library contains no mass-compatible molecule',()=>{
  assert.deepEqual(search(data.library,[[100,1]],{precursor:501}).ranked,[]);
});
test('adduct correction preserves neutral mass when the precursor is shifted correctly',()=>{
  const e=data.examples[0];const shifted=e.precursor-ADDUCTS['[M+H]+']+ADDUCTS['[M+Na]+'];
  assert.equal(search(data.library,e.peaks,{precursor:shifted,adduct:'[M+Na]+'}).ranked[0].id,e.compoundId);
});
test('peak input accepts headers, CSV, whitespace, comments and arbitrary intensity scale',()=>{
  assert.deepEqual(parsePeaks('m/z, intensity\n# spectrum\n100, 200\n120\t100'),[[100,1],[120,.5]]);
  assert.deepEqual(parsePeaks('100, 2\n120, 1'),parsePeaks('100, 200\n120, 100'));
});
test('invalid or empty inputs produce actionable errors',()=>{
  for(const input of ['', '100,-1','-3,5','100,NaN','100,0','1,2,3','Infinity,3'])assert.throws(()=>parsePeaks(input));
  for(const precursor of [0,-1,NaN,Infinity])assert.throws(()=>search(data.library,[[100,1]],{precursor}));
  assert.throws(()=>search(data.library,[[100,1]],{precursor:200,ppm:0}));
  assert.throws(()=>search(data.library,[[1,1]],{precursor:200}));
});
test('normalized identical spectra score one; disjoint spectra score zero',()=>{
  const q=vectorize([[100,1],[120,.5]]);assert.ok(Math.abs(cosine(q,q).score-1)<1e-12);
  assert.equal(cosine(q,vectorize([[200,1]])).score,0);
});
test('neutral losses use precursor minus fragment and do not invent absent evidence',()=>{
  const match=neutralLosses([[200-18.010565,1]],200);
  assert.equal(match[0].formula,'H₂O');assert.deepEqual(neutralLosses([[100,1]],200),[]);
});
test('references and queries have distinct accessions; provenance and originals are present',()=>{
  const references=data.library.flatMap(c=>c.spectra),ids=new Set(references.map(s=>s.id));
  assert.equal(ids.size,8);
  for(const r of [...references,...data.examples]){
    assert.equal(r.license,'CC0');assert.ok(r.source.includes(data.pin));
    const raw=fs.readFileSync(new URL('../dist/'+r.raw,import.meta.url),'utf8');
    assert.ok(raw.includes('LICENSE: CC0'));assert.ok(raw.includes('ACCESSION: '+r.id));
  }
  for(const q of data.examples)assert.equal(ids.has(q.id),false);
});
