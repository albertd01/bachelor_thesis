from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import hashlib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

def hash_atom_info(atom):
    features = (
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetNumRadicalElectrons(),
        str(atom.GetHybridization()),
        atom.GetIsAromatic(),
        atom.IsInRing()
    )
    s = str(features)
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h, 16) & 0xFFFFFFFF  # 32-bit

def hash_tuple(t):
    h = hashlib.sha1(str(t).encode('utf-8')).hexdigest()
    return int(h, 16) & 0xFFFFFFFF


# From scratch implemenetation
def generate_ecfp(mol, radius=2, nBits=2048):
    fp_bits = set()
    atom_ids = {}
    
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        atom_ids[idx] = hash_atom_info(atom)

    for iteration in range(radius):
        new_ids = {}
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            info = (atom_ids[idx],) + tuple(
                atom_ids[neighbor.GetIdx()]
                for neighbor in atom.GetNeighbors()
            )
            env_id = hash_tuple(info)
            fp_bits.add(env_id % nBits)
            new_ids[idx] = env_id
        atom_ids = new_ids

    fp_array = np.zeros(nBits, dtype=int)
    for bit in fp_bits:
        fp_array[bit] = 1
    return fp_array

def get_custom_invariants(mol):
    """Compute a list of integer invariants, one per atom in the molecule,
    using a tuple of properties. Adjust the tuple to include the properties you need."""
    invariants = []
    for atom in mol.GetAtoms():
        invariants.append(hash_atom_info(atom))
    return invariants

# RDKit implementation
def compute_ecfp_array(smiles: str, radius: int, nBits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    invariants = get_custom_invariants(mol)  
    
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol, customAtomInvariants=invariants)
    info = {}
    print("RDKit bitInfo:")
    for bit, entries in info.items():
        print(f"Bit {bit}: {entries}")

    arr = np.zeros((nBits,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


from rdkit.Chem import MolFromSmiles

smiles = 'CCO' 
mol = MolFromSmiles(smiles)
fingerprint_from_scratch = generate_ecfp(mol, radius=2, nBits=1024)
fingerprint_rdkit = compute_ecfp_array(smiles, radius=2, nBits=1024)

print("From scratch bits:", np.where(fingerprint_from_scratch == 1)[0])
print("RDKit bits:", np.where(fingerprint_rdkit == 1)[0])

intersection = np.sum(np.logical_and(fingerprint_from_scratch, fingerprint_rdkit))
union = np.sum(np.logical_or(fingerprint_from_scratch, fingerprint_rdkit))
tanimoto_similarity = intersection / union
print(f"Tanimoto similarity: {tanimoto_similarity:.4f}")