"""
api.py — Backend FastAPI para o Molecule Lab
=============================================

Expõe um único endpoint:

    POST /api/molecule/analyze
    { "graph": { "atoms": [...], "bonds": [...] } }

Fluxo interno:
    grafo JSON  →  RDKit RWMol  →  SMILES  →  run_md()  →  resposta JSON

Executar:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import math
import json
import logging
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem

import time

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(**kwargs):          # type: ignore[misc]
        def decorator(f): return f
        return decorator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("molecule-api")

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Molecule Lab API")

app.add_middleware(
    CORSMiddleware,
    # em produção, restrinja ao origin do frontend
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Modelos de entrada
# ---------------------------------------------------------------------------


class GraphAtom(BaseModel):
    id: str
    symbol: str
    x: float
    y: float


class GraphBond(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    order: int = 1

    class Config:
        populate_by_name = True


class MoleculeGraph(BaseModel):
    atoms: list[GraphAtom]
    bonds: list[GraphBond]


class AnalyzeRequest(BaseModel):
    graph: MoleculeGraph

# ---------------------------------------------------------------------------
# Conversão grafo → RDKit
# ---------------------------------------------------------------------------


_BOND_TYPE_FROM_ORDER = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def graph_to_smiles(graph: MoleculeGraph) -> str:
    """Constrói um RWMol a partir do grafo do canvas e retorna o SMILES canônico."""
    mol = Chem.RWMol()
    id_to_idx: dict[str, int] = {}

    for atom in graph.atoms:
        idx = mol.AddAtom(Chem.Atom(atom.symbol))
        id_to_idx[atom.id] = idx

    for bond in graph.bonds:
        btype = _BOND_TYPE_FROM_ORDER.get(bond.order, Chem.BondType.SINGLE)
        mol.AddBond(id_to_idx[bond.from_], id_to_idx[bond.to], btype)

    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)

# ---------------------------------------------------------------------------
# Parâmetros globais da simulação
# (idênticos ao notebook; altere aqui para toda a aplicação)
# ---------------------------------------------------------------------------


TEMPERATURE_START = 1.0
TEMPERATURE_END = 10_000.0
N_STEPS = 15_000
DT = 3.0e-4
GAMMA = 20.0
RESET_ANG_MOM = 200
BREAK_FRAC = 0.95
BREAK_PERSISTENCE = 20
R_CUT = 8.0
R_SKIN = 1.5
K_COULOMB = 50.0
LJ_SCALE_14 = 0.5
SHAKE_TOL = 1e-8
SHAKE_MAX_ITER = 50
SAVE_EVERY = 5

BOND_K_MAP = {
    Chem.rdchem.BondType.SINGLE:   300.0,
    Chem.rdchem.BondType.DOUBLE:   500.0,
    Chem.rdchem.BondType.TRIPLE:   700.0,
    Chem.rdchem.BondType.AROMATIC: 450.0,
}
BOND_DISS_MAP = {
    Chem.rdchem.BondType.SINGLE:   150.0,
    Chem.rdchem.BondType.DOUBLE:   280.0,
    Chem.rdchem.BondType.TRIPLE:   420.0,
    Chem.rdchem.BondType.AROMATIC: 210.0,
}
DIHEDRAL_PARAMS = {
    Chem.rdchem.BondType.SINGLE:   (0.0,  0.0,  1.5),
    Chem.rdchem.BondType.DOUBLE:   (0.0, 15.0,  0.0),
    Chem.rdchem.BondType.AROMATIC: (0.0, 10.0,  0.0),
    Chem.rdchem.BondType.TRIPLE:   (0.0,  0.0,  0.0),
}
ANGLE_K_HYB = {
    Chem.rdchem.HybridizationType.SP3:   100.0,
    Chem.rdchem.HybridizationType.SP2:   140.0,
    Chem.rdchem.HybridizationType.SP:    200.0,
    Chem.rdchem.HybridizationType.OTHER: 100.0,
}
EPSILON_MAP = {'H': 0.02, 'C': 0.08, 'N': 0.10, 'O': 0.12, 'S': 0.15,
               'F': 0.10, 'Cl': 0.15, 'Br': 0.20, 'I': 0.25}
SIGMA_SCALE = 0.85

# ---------------------------------------------------------------------------
# Construção e topologia
# ---------------------------------------------------------------------------


def build_molecule(smiles: str, seed: int = 0xF00D):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError("SMILES inválido.")
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDGv3 falhou.")
    AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def get_partial_charges(mol) -> np.ndarray:
    AllChem.ComputeGasteigerCharges(mol)
    charges = []
    for i in range(mol.GetNumAtoms()):
        q = mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")
        charges.append(q if math.isfinite(q) else 0.0)
    return np.array(charges, dtype=float)


def set_positions(mol, pos: np.ndarray):
    conf = mol.GetConformer()
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, p.tolist())


def lj_params(si, sj, ri, rj):
    sigma = SIGMA_SCALE * (ri + rj)
    epsilon = math.sqrt(EPSILON_MAP.get(si, 0.08) * EPSILON_MAP.get(sj, 0.08))
    return sigma, epsilon


def build_topology(mol):
    conf = mol.GetConformer()
    pt = Chem.GetPeriodicTable()
    atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())]
    masses = np.array([a.GetMass() for a in atoms])
    radii = np.array([pt.GetRcovalent(a.GetAtomicNum()) for a in atoms])
    symbols = [a.GetSymbol() for a in atoms]
    hybs = [a.GetHybridization() for a in atoms]
    charges = get_partial_charges(mol)

    pos = np.array([[conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(mol.GetNumAtoms())], dtype=float)

    bonds = []
    neighbors = {i: set() for i in range(mol.GetNumAtoms())}
    bond_bt_map = {}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        r0 = np.linalg.norm(pos[j] - pos[i])
        kh = BOND_K_MAP.get(bond.GetBondType(), 300.0)
        De = BOND_DISS_MAP.get(bond.GetBondType(), 150.0)
        alpha = math.sqrt(kh / (2.0 * De))
        bonds.append((i, j, r0, De, alpha))
        neighbors[i].add(j)
        neighbors[j].add(i)
        bond_bt_map[(min(i, j), max(i, j))] = bond.GetBondType()

    bonded_pairs = {(min(i, j), max(i, j)) for i, j, *_ in bonds}

    one_three = set()
    for c in range(mol.GetNumAtoms()):
        nbrs = sorted(neighbors[c])
        for a in range(len(nbrs)):
            for b in range(a+1, len(nbrs)):
                one_three.add((min(nbrs[a], nbrs[b]), max(nbrs[a], nbrs[b])))

    angles = []
    for c in range(mol.GetNumAtoms()):
        nbrs = sorted(neighbors[c])
        k_theta = ANGLE_K_HYB.get(hybs[c], 100.0)
        for a in range(len(nbrs)):
            for b in range(a+1, len(nbrs)):
                ii, jj = nbrs[a], nbrs[b]
                rci = pos[ii] - pos[c]
                rcj = pos[jj] - pos[c]
                r_ci = np.linalg.norm(rci)+1e-12
                r_cj = np.linalg.norm(rcj)+1e-12
                cos_t = np.clip(np.dot(rci, rcj)/(r_ci*r_cj), -1.0, 1.0)
                angles.append((ii, c, jj, math.acos(cos_t), k_theta))

    dihedrals = []
    seen_dih = set()
    for (p, q), bt in bond_bt_map.items():
        V1, V2, V3 = DIHEDRAL_PARAMS.get(bt, (0.0, 0.0, 1.5))
        if V1 == V2 == V3 == 0.0:
            continue
        for i in neighbors[p]:
            if i == q:
                continue
            for l in neighbors[q]:
                if l == p or l == i:
                    continue
                tup = (i, p, q, l)
                key = min(tup, (l, q, p, i))
                if key in seen_dih:
                    continue
                seen_dih.add(key)
                dihedrals.append((i, p, q, l, V1, V2, V3))

    one_four = set()
    for i, j, k, l, *_ in dihedrals:
        pair = (min(i, l), max(i, l))
        if pair not in bonded_pairs and pair not in one_three:
            one_four.add(pair)

    shake_bonds = [(i, j, r0) for i, j, r0, De, alpha in bonds
                   if symbols[i] == 'H' or symbols[j] == 'H']

    return (pos, masses, radii, symbols, charges,
            bonds, bonded_pairs, one_three, one_four,
            angles, dihedrals, shake_bonds)

# ---------------------------------------------------------------------------
# SHAKE / RATTLE
# ---------------------------------------------------------------------------


def shake_positions(pos, masses, shake_bonds, tol=SHAKE_TOL, max_iter=SHAKE_MAX_ITER):
    for _ in range(max_iter):
        converged = True
        for i, j, d0 in shake_bonds:
            rij = pos[j] - pos[i]
            d2 = float(np.dot(rij, rij))
            d0_2 = d0*d0
            if abs(d2 - d0_2) < tol * d0_2:
                continue
            converged = False
            lam = (d2 - d0_2) / (2.0 * (1/masses[i] + 1/masses[j]) * d2)
            pos[i] += lam / masses[i] * rij
            pos[j] -= lam / masses[j] * rij
        if converged:
            break
    return pos


def rattle_velocities(pos, vel, masses, shake_bonds, tol=SHAKE_TOL, max_iter=SHAKE_MAX_ITER):
    for _ in range(max_iter):
        converged = True
        for i, j, d0 in shake_bonds:
            rij = pos[j] - pos[i]
            vij = vel[j] - vel[i]
            rv = float(np.dot(rij, vij))
            if abs(rv) < tol:
                continue
            converged = False
            mu = rv / ((1/masses[i] + 1/masses[j]) * d0*d0)
            vel[i] += mu / masses[i] * rij
            vel[j] -= mu / masses[j] * rij
        if converged:
            break
    return vel

# ---------------------------------------------------------------------------
# BAOAB
# ---------------------------------------------------------------------------


def baoab_step(pos, vel, masses, forces, dt, gamma, target_T,
               rng, shake_bonds, force_fn):
    acc = forces / masses[:, None]
    vel = vel + 0.5*dt*acc
    pos = pos + 0.5*dt*vel
    if shake_bonds:
        pos = shake_positions(pos, masses, shake_bonds)
    c = math.exp(-gamma * dt)
    noise_std = np.sqrt(max(0.0, 1-c*c) * target_T / masses)[:, None]
    vel = c * vel + noise_std * rng.standard_normal(vel.shape)
    pos = pos + 0.5*dt*vel
    if shake_bonds:
        pos = shake_positions(pos, masses, shake_bonds)
    new_forces, potential, atom_energies = force_fn(pos)
    vel = vel + 0.5*dt * new_forces / masses[:, None]
    if shake_bonds:
        vel = rattle_velocities(pos, vel, masses, shake_bonds)
    return pos, vel, new_forces, potential, atom_energies

# ---------------------------------------------------------------------------
# Temperatura e momento
# ---------------------------------------------------------------------------


def initialize_velocities(masses, temperature, seed=123):
    rng = np.random.default_rng(seed)
    vel = rng.normal(size=(len(masses), 3)) * np.sqrt(0.05 *
                                                      max(temperature, 1e-6)/masses)[:, None]
    vel -= np.average(vel, axis=0, weights=masses)
    return vel


def kinetic_temperature(vel, masses, Nf=None):
    K = 0.5 * np.sum(masses[:, None] * vel**2)
    Nf = max(1, 3*len(masses)-6) if Nf is None else Nf
    return 2.0*K/Nf, K, Nf


def temperature_ramp(step, n_steps, t0, t1):
    return t0 + (t1-t0)*step/max(n_steps-1, 1)


def remove_linear_momentum(vel, masses):
    vel -= np.average(vel, axis=0, weights=masses)
    return vel


def remove_angular_momentum(pos, vel, masses):
    M = masses.sum()
    r_com = (masses[:, None]*pos).sum(0)/M
    r_rel = pos - r_com
    L = (masses[:, None]*np.cross(r_rel, vel)).sum(0)
    I = np.zeros((3, 3))
    for k in range(len(masses)):
        r = r_rel[k]
        I += masses[k]*(np.dot(r, r)*np.eye(3) - np.outer(r, r))
    try:
        omega = np.linalg.solve(I, L)
        vel -= np.cross(omega[None, :], r_rel)
    except np.linalg.LinAlgError:
        pass
    return vel

# ---------------------------------------------------------------------------
# Verlet list
# ---------------------------------------------------------------------------


class VerletList:
    def __init__(self, r_cut, r_skin):
        self.r_cut = r_cut
        self.r_skin = r_skin
        self.pos_ref = None
        self.pair_i = self.pair_j = np.empty(0, dtype=np.int32)
        self.epsilons = self.sigmas = self.scales = np.empty(0)
        self.charges_i = self.charges_j = np.empty(0)

    def needs_rebuild(self, pos):
        if self.pos_ref is None:
            return True
        return float(np.max(np.linalg.norm(pos - self.pos_ref, axis=1))) > self.r_skin*0.5

    def build(self, pos, bonded_pairs, one_three, one_four, radii, symbols, charges):
        r_list = self.r_cut + self.r_skin
        pi_l = []
        pj_l = []
        eps_l = []
        sig_l = []
        sc_l = []
        qi_l = []
        qj_l = []
        n = len(pos)
        for i in range(n):
            for j in range(i+1, n):
                pair = (i, j)
                if pair in bonded_pairs or pair in one_three:
                    continue
                if np.linalg.norm(pos[j]-pos[i]) > r_list:
                    continue
                scale = LJ_SCALE_14 if pair in one_four else 1.0
                sigma, epsilon = lj_params(
                    symbols[i], symbols[j], radii[i], radii[j])
                pi_l.append(i)
                pj_l.append(j)
                eps_l.append(epsilon)
                sig_l.append(sigma)
                sc_l.append(scale)
                qi_l.append(charges[i])
                qj_l.append(charges[j])
        self.pair_i = np.array(pi_l, dtype=np.int32)
        self.pair_j = np.array(pj_l, dtype=np.int32)
        self.epsilons = np.array(eps_l)
        self.sigmas = np.array(sig_l)
        self.scales = np.array(sc_l)
        self.charges_i = np.array(qi_l)
        self.charges_j = np.array(qj_l)
        self.pos_ref = pos.copy()

# ---------------------------------------------------------------------------
# Kernel não-ligante (Numba / Python puro)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _nonbonded_kernel(pos, pair_i, pair_j, epsilons, sigmas, scales,
                      charges_i, charges_j, k_coulomb, r_cut):
    n = pos.shape[0]
    forces = np.zeros((n, 3))
    energy = 0.0
    ae = np.zeros(n)
    for k in range(len(pair_i)):
        i = pair_i[k]
        j = pair_j[k]
        sc = scales[k]
        dx = pos[j, 0]-pos[i, 0]
        dy = pos[j, 1]-pos[i, 1]
        dz = pos[j, 2]-pos[i, 2]
        r2 = dx*dx+dy*dy+dz*dz+1e-24
        r = math.sqrt(r2)
        if r > r_cut:
            continue
        sig = sigmas[k]
        eps = epsilons[k]
        sr2 = (sig/r)**2
        sr6 = sr2*sr2*sr2
        sr12 = sr6*sr6
        v_lj = 4.0*eps*(sr12-sr6)*sc
        f_lj = 24.0*eps/r*(2.0*sr12-sr6)*sc
        qi = charges_i[k]
        qj = charges_j[k]
        v_c = k_coulomb*qi*qj/r*sc
        f_c = k_coulomb*qi*qj/r2*sc
        f_tot = f_lj+f_c
        fx = f_tot*dx/r
        fy = f_tot*dy/r
        fz = f_tot*dz/r
        forces[i, 0] -= fx
        forces[i, 1] -= fy
        forces[i, 2] -= fz
        forces[j, 0] += fx
        forces[j, 1] += fy
        forces[j, 2] += fz
        half_e = 0.5*(v_lj+v_c)
        ae[i] += half_e
        ae[j] += half_e
        energy += v_lj+v_c
    return forces, energy, ae


def nonbonded_forces(pos, vlist):
    if len(vlist.pair_i) == 0:
        n = len(pos)
        return np.zeros((n, 3)), 0.0, np.zeros(n)
    return _nonbonded_kernel(pos, vlist.pair_i, vlist.pair_j,
                             vlist.epsilons, vlist.sigmas, vlist.scales,
                             vlist.charges_i, vlist.charges_j, K_COULOMB, R_CUT)

# ---------------------------------------------------------------------------
# Forças ligantes
# ---------------------------------------------------------------------------


def bonded_forces_and_energy(pos, bonds, angles, dihedrals):
    n = len(pos)
    forces = np.zeros((n, 3))
    energy = 0.0
    ae = np.zeros(n)
    for i, j, r0, De, alpha in bonds:
        rij = pos[j]-pos[i]
        r = np.linalg.norm(rij)+1e-12
        u = math.exp(-alpha*(r-r0))
        v = De*(1-u)**2
        dVdr = 2*De*alpha*u*(1-u)
        fij = dVdr*rij/r
        forces[i] += fij
        forces[j] -= fij
        energy += v
        ae[i] += 0.5*v
        ae[j] += 0.5*v
    for i, c, j, theta0, k_theta in angles:
        rci = pos[i]-pos[c]
        rcj = pos[j]-pos[c]
        r_ci = np.linalg.norm(rci)+1e-12
        r_cj = np.linalg.norm(rcj)+1e-12
        cos_t = np.clip(np.dot(rci, rcj)/(r_ci*r_cj), -1+1e-9, 1-1e-9)
        theta = math.acos(cos_t)
        sin_t = math.sqrt(max(1-cos_t**2, 1e-12))
        dth = theta-theta0
        v = 0.5*k_theta*dth**2
        coeff = -k_theta*dth/sin_t
        fi = coeff/r_ci*(rcj/r_cj-cos_t*rci/r_ci)
        fj = coeff/r_cj*(rci/r_ci-cos_t*rcj/r_cj)
        forces[i] += fi
        forces[j] += fj
        forces[c] -= (fi+fj)
        energy += v
        ae[i] += v/3
        ae[c] += v/3
        ae[j] += v/3
    for i, j, k, l, V1, V2, V3 in dihedrals:
        b1 = pos[j]-pos[i]
        b2 = pos[k]-pos[j]
        b3 = pos[l]-pos[k]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        n1n = np.linalg.norm(n1)+1e-12
        n2n = np.linalg.norm(n2)+1e-12
        b2n = np.linalg.norm(b2)+1e-12
        n1h = n1/n1n
        b2h = b2/b2n
        m1 = np.cross(n1h, b2h)
        cos_phi = np.clip(np.dot(n1h, n2/n2n), -1+1e-9, 1-1e-9)
        sin_phi = np.dot(m1, n2/n2n)
        phi = math.atan2(sin_phi, cos_phi)
        v = V1/2*(1+math.cos(phi))+V2/2 * \
            (1-math.cos(2*phi))+V3/2*(1+math.cos(3*phi))
        dVdp = -V1/2*math.sin(phi)+V2*math.sin(2*phi)-3*V3/2*math.sin(3*phi)
        energy += v
        ae[i] += v/4
        ae[j] += v/4
        ae[k] += v/4
        ae[l] += v/4
        if abs(dVdp) < 1e-12:
            continue
        fi = (dVdp*b2n/n1n**2)*n1
        fl = -(dVdp*b2n/n2n**2)*n2
        b2l2 = b2n**2
        fj = (np.dot(b1, b2)/b2l2-1)*fi-np.dot(b3, b2)/b2l2*fl
        fk = -(fi+fj+fl)
        forces[i] += fi
        forces[j] += fj
        forces[k] += fk
        forces[l] += fl
    return forces, energy, ae


def forces_and_energy(pos, bonds, angles, dihedrals, vlist):
    fb, eb, aeb = bonded_forces_and_energy(pos, bonds, angles, dihedrals)
    fn, en, aen = nonbonded_forces(pos, vlist)
    return fb+fn, eb+en, aeb+aen

# ---------------------------------------------------------------------------
# Detecção de quebra
# ---------------------------------------------------------------------------


def detect_broken_bonds(pos, bonds, break_frac=BREAK_FRAC):
    broken = []
    for idx, (i, j, r0, De, alpha) in enumerate(bonds):
        r = np.linalg.norm(pos[i]-pos[j])
        u = math.exp(-alpha*(r-r0))
        V = De*(1-u)**2
        if V > break_frac*De and r > r0:
            broken.append({"bond_index": idx, "i": i, "j": j,
                           "distance": float(r), "r0": float(r0),
                           "V": float(V), "De": float(De)})
    return broken

# ---------------------------------------------------------------------------
# Loop de MD
# ---------------------------------------------------------------------------


def run_md(smiles, **kwargs):
    params = dict(
        temperature_start=TEMPERATURE_START, temperature_end=TEMPERATURE_END,
        n_steps=N_STEPS, dt=DT, gamma=GAMMA,
        break_persistence=BREAK_PERSISTENCE, break_frac=BREAK_FRAC,
        save_every=SAVE_EVERY, reset_ang_mom=RESET_ANG_MOM,
    )
    params.update(kwargs)

    mol = build_molecule(smiles)
    (pos, masses, radii, symbols, charges,
     bonds, bonded_pairs, one_three, one_four,
     angles, dihedrals, shake_bonds) = build_topology(mol)

    vlist = VerletList(R_CUT, R_SKIN)
    vlist.build(pos, bonded_pairs, one_three,
                one_four, radii, symbols, charges)

    vel = initialize_velocities(masses, params["temperature_start"])
    vel = remove_angular_momentum(pos, vel, masses)
    forces, potential, atom_e = forces_and_energy(
        pos, bonds, angles, dihedrals, vlist)

    rng = np.random.default_rng(42)

    def force_fn(p):
        if vlist.needs_rebuild(p):
            vlist.build(p, bonded_pairs, one_three,
                        one_four, radii, symbols, charges)
        return forces_and_energy(p, bonds, angles, dihedrals, vlist)

    temperatures = []
    target_temps = []
    persist_cnt = {idx: 0 for idx in range(len(bonds))}

    for step in range(params["n_steps"]):
        target_T = temperature_ramp(step, params["n_steps"],
                                    params["temperature_start"],
                                    params["temperature_end"])
        pos, vel, forces, potential, atom_e = baoab_step(
            pos, vel, masses, forces, params["dt"], params["gamma"],
            target_T, rng, shake_bonds, force_fn
        )
        vel = remove_linear_momentum(vel, masses)
        if params["reset_ang_mom"] > 0 and (step+1) % params["reset_ang_mom"] == 0:
            vel = remove_angular_momentum(pos, vel, masses)

        T_curr, _, _ = kinetic_temperature(vel, masses)
        temperatures.append(T_curr)
        target_temps.append(target_T)

        broken_now = detect_broken_bonds(pos, bonds, params["break_frac"])
        active = {e["bond_index"] for e in broken_now}
        for idx in persist_cnt:
            persist_cnt[idx] = persist_cnt[idx]+1 if idx in active else 0
        persistent = [e for e in broken_now
                      if persist_cnt[e["bond_index"]] >= params["break_persistence"]]

        if persistent:
            return {
                "result":            "break",
                "break_step":        step,
                "break_temperature": float(target_T),
                "broken_bonds":      persistent,
                "symbols":           symbols,
                "temperatures":      temperatures,
                "target_temperatures": target_temps,
            }

    return {
        "result":            "stable",
        "break_step":        None,
        "break_temperature": None,
        "broken_bonds":      [],
        "symbols":           symbols,
        "temperatures":      temperatures,
        "target_temperatures": target_temps,
    }

# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@app.post("/api/molecule/analyze")
def analyze(req: AnalyzeRequest):
    start_time = time.time()

    graph = req.graph

    # 1. Grafo → SMILES
    try:
        smiles = graph_to_smiles(graph)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Molécula inválida: {e}")

    new_time = time.time() - start_time
    log.info(f" {new_time:.2f}s SMILES gerado: {smiles}")
    start_time += new_time

    # 2. Propriedades estáticas (RDKit)
    mol_static = Chem.MolFromSmiles(smiles)
    formula = rdMolDescriptors.CalcMolFormula(mol_static)
    mw = Descriptors.MolWt(mol_static)
    num_rings = mol_static.GetRingInfo().NumRings()
    is_arom = any(a.GetIsAromatic() for a in mol_static.GetAtoms())

    new_time = time.time() - start_time
    log.info(f" {new_time:.2f}s Carregou propriedades estáticas (RDKit)")
    start_time += new_time

    # 3. Simulação de dinâmica molecular
    try:
        sim = run_md(smiles)
    except Exception as e:
        log.exception("run_md falhou")
        raise HTTPException(status_code=500, detail=f"Erro na simulação: {e}")

    new_time = time.time() - start_time
    log.info(f" {new_time:.2f}s Simulação de dinâmica molecular")
    start_time += new_time

    # 4. Formata bonds quebradas para o frontend
    broken_out = []
    for b in sim["broken_bonds"]:
        si = sim["symbols"][b["i"]]
        sj = sim["symbols"][b["j"]]
        broken_out.append({
            "atom_i": si, "atom_j": sj,
            "distance": round(b["distance"], 3),
            "r0":       round(b["r0"], 3),
            "fraction": round(b["V"] / b["De"], 3),
        })

    new_time = time.time() - start_time
    log.info(f" {new_time:.2f}s Formatação de ligações quebradas")
    start_time += new_time

    return {
        "smiles":            smiles,
        "formula":           formula,
        "name":              None,
        "molecular_weight":  round(mw, 4),
        "valid":             True,
        "result":            sim["result"],          # "stable" | "break"
        "break_temperature": sim["break_temperature"],  # float em K, ou null
        "broken_bonds":      broken_out,
        "properties": {
            "num_atoms":   mol_static.GetNumAtoms(),
            "num_bonds":   mol_static.GetNumBonds(),
            "num_rings":   num_rings,
            "is_aromatic": is_arom,
        },
        "error": None,
    }


@app.get('/hello')
def hello():
    return "API funcionando corretamente!"
