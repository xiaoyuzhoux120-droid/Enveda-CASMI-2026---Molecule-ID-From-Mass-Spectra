"""Build a small, attributed browser library from pinned CC0 MassBank records."""
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]
PIN = 'c1c272f44837881d2f310a8171739a6c2223408f'
QUERIES = {'Kaempferol': 'MSBNK-NaToxAq-NA003753', 'Daidzein': 'MSBNK-NaToxAq-NA003635',
           '8-Prenylnaringenin': 'MSBNK-NaToxAq-NA003223', 'Genistein': 'MSBNK-UFZ-WANA0496213166PH'}
records = []
for path in sorted((ROOT / 'dist/data/records').glob('*.txt')):
    text = path.read_text()
    def field(prefix):
        return next(line[len(prefix):].strip() for line in text.splitlines() if line.startswith(prefix))
    assert field('LICENSE:') == 'CC0'
    accession = field('ACCESSION:')
    folder = 'UFZ' if accession.startswith('MSBNK-UFZ') else 'NaToxAq'
    peaks = []
    for line in text.split('PK$PEAK: m/z int. rel.int.\n')[1].splitlines():
        if line.startswith('//'): break
        values = line.split()
        if len(values) == 3: peaks.append([float(values[0]), float(values[1])])
    highest = max(i for m,i in peaks)
    records.append(dict(id=accession, name=field('CH$NAME:'), formula=field('CH$FORMULA:'),
                        smiles=field('CH$SMILES:'), precursor=float(field('MS$FOCUSED_ION: PRECURSOR_M/Z')),
                        adduct=field('MS$FOCUSED_ION: PRECURSOR_TYPE'),
                        energy=field('AC$MASS_SPECTROMETRY: COLLISION_ENERGY'),
                        peaks=[[m, round(i/highest, 9)] for m,i in peaks], authors=field('AUTHORS:'),
                        license='CC0', source=f'https://github.com/MassBank/MassBank-data/blob/{PIN}/{folder}/{path.name}',
                        raw=f'data/records/{path.name}'))
library, examples = [], []
for name in ['Kaempferol', 'Genistein', 'Daidzein', '8-Prenylnaringenin']:
    rows = [r for r in records if r['name'] == name]
    query = next(r for r in rows if r['id'] == QUERIES[name])
    reference = [r for r in rows if r['id'] != query['id']]
    molecule = Chem.MolFromSmiles(query['smiles'])
    assert molecule and len(reference) == 2
    slug = name.lower().replace('8-', '').replace(' ', '-')
    drawer = rdMolDraw2D.MolDraw2DSVG(360, 210)
    drawer.drawOptions().clearBackground = False
    drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    (ROOT / f'dist/assets/{slug}.svg').write_text(drawer.GetDrawingText())
    library.append(dict(id=slug, name=name, formula=query['formula'], smiles=query['smiles'],
                        mass=rdMolDescriptors.CalcExactMolWt(molecule), image=f'assets/{slug}.svg',
                        spectra=reference))
    examples.append(dict(**query, compoundId=slug))
output = dict(version=1, pin=PIN, description='CC0 MassBank demonstration subset; no CASMI competition data.',
              library=library, examples=examples)
(ROOT / 'dist/data/library.json').write_text(json.dumps(output, indent=2) + '\n')
print(f'Built {len(library)} compounds, {sum(len(c["spectra"]) for c in library)} reference spectra, {len(examples)} separate queries.')
