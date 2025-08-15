import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray

# Settings
RADIUS = 2
N_BITS = 2048
CSV_PATH = "train_full.csv"  # adjust path
OUT_PATH = "train_morgan_features_2048.npy"

# Load SMILES
df = pd.read_csv(CSV_PATH)

# Generate Morgan fingerprints
fps = []
for smi in df["SMILES"]:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, nBits=N_BITS)
        arr = np.zeros((N_BITS,), dtype=int)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)
    else:
        fps.append(np.zeros((N_BITS,), dtype=int))

fps = np.array(fps, dtype=bool)
np.save(OUT_PATH, fps)

print(f"Saved {fps.shape} to {OUT_PATH}")
