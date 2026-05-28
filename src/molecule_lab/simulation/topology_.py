#!/usr/bin/env python3
"""Construção da molécula e extração da topologia (RDKit)."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .parameters_ import BOND_K


def build_molecule(smiles: str, seed: int = 0xF00D):
    """Constrói molécula 3D com RDKit (ETKDG + UFF)."""
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError(f"SMILES inválido: {smiles!r}")
    mol = Chem.AddHs(base)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise RuntimeError("Falha ao gerar geometria 3D.")
    AllChem.UFFOptimizeMolecule(mol)
    return mol


def bond_type_str(bond) -> str:
    """Converte BondType RDKit para string."""
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:   return "SINGLE"
    if bt == Chem.rdchem.BondType.DOUBLE:   return "DOUBLE"
    if bt == Chem.rdchem.BondType.TRIPLE:   return "TRIPLE"
    if bt == Chem.rdchem.BondType.AROMATIC: return "AROMATIC"
    return "SINGLE"


def build_topology(mol):
    """
    Extrai posições, massas, raios covalentes, símbolos, ligações,
    e pares 1-3 (excluídos da repulsão não‑ligante).
    """
    conf = mol.GetConformer()
    pt   = Chem.GetPeriodicTable()
    n    = mol.GetNumAtoms()
    atoms = [mol.GetAtomWithIdx(i) for i in range(n)]

    symbols = [a.GetSymbol() for a in atoms]
    masses  = np.array([a.GetMass() for a in atoms], dtype=float)
    radii   = np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in atoms], dtype=float)
    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(n)], dtype=float)

    bonds = []
    neighbors = {i: set() for i in range(n)}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0 = float(np.linalg.norm(pos[j] - pos[i]))
        bt = bond_type_str(bond)
        k = BOND_K.get(bt, 300.0)
        bonds.append((i, j, r0, k, bt))
        neighbors[i].add(j)
        neighbors[j].add(i)

    # Pares 1-3 (compartilham um vizinho)
    one_three = set()
    for center in range(n):
        nbrs = sorted(neighbors[center])
        for a in range(len(nbrs)):
            for b in range(a+1, len(nbrs)):
                one_three.add(tuple(sorted((nbrs[a], nbrs[b]))))

    # Pares ligados diretamente (1-2)
    bonded_pairs = {tuple(sorted((i, j))) for i, j, *_ in bonds}

    return pos, masses, radii, symbols, bonds, neighbors, one_three, bonded_pairs


def set_positions(mol, pos: np.ndarray) -> None:
    """Atualiza as posições no objeto RDKit."""
    conf = mol.GetConformer()
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, p)