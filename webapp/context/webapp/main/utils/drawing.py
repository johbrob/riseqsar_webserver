from rdkit import Chem
import rdkit.Chem.Draw as Draw
import os
import io
import base64

def draw_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol_img = Draw.MolToImage(mol)
    return mol_img

def draw_n_save_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    print('entering')
    print(os.getcwd())
    a = Draw.MolToFile(mol, 'webapp/tmp/mol.png')
    from pathlib import Path
    print(Path('webapp/tmp/mol.png').is_file())


def PIL2b64(pil_img):
    image_io = io.BytesIO()
    pil_img.save(image_io, format='PNG')
    encoded = base64.b64encode(image_io.getvalue()).decode("utf-8")
    return encoded

def smiles_2_b64_img(smiles):
    return PIL2b64(draw_mol(smiles))
