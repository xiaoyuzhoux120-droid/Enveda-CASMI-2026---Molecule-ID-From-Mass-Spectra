"""Official structural target helpers without a GPU runtime dependency."""
import hashlib

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize


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


def validated_pool_target(smiles, expected_identity, enumerator=None):
    """Exact pool fingerprint path with a reusable tautomer enumerator."""
    return canonical_target(smiles, expected_identity, enumerator)


def canonical_record(smiles):
    """Dataset preparation only, without any query or label input."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms() or len(Chem.GetMolFrags(mol)) != 1:
        return None
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    return Chem.MolToInchiKey(mol)[:14], Chem.MolToSmiles(mol)
