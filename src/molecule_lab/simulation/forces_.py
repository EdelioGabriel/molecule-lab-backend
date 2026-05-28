#!/usr/bin/env python3
"""Cálculo de forças (ligações harmônicas + WCA)."""

import math
import numpy as np

from .parameters_ import EPSILON_MAP, SIGMA_SCALE


def lj_params(symbol_i, symbol_j, radius_i, radius_j):
    """Parâmetros WCA: sigma = 0.85*(ri+rj), epsilon = sqrt(εi εj)."""
    sigma = SIGMA_SCALE * (radius_i + radius_j)
    epsilon = math.sqrt(EPSILON_MAP.get(symbol_i, 0.08) *
                        EPSILON_MAP.get(symbol_j, 0.08))
    return sigma, epsilon


def forces_and_energy(pos, radii, symbols, bonds, one_three, bonded_pairs):
    """
    Calcula forças e energia potencial total.
    Ligações: mola harmônica  F = k (r - r0) r̂
    Não‑ligantes: WCA (repulsão LJ truncada) excluindo 1-2 e 1-3.
    """
    n = len(pos)
    forces = np.zeros_like(pos)
    energy = 0.0

    # Ligações
    for i, j, r0, k, _ in bonds:
        rij = pos[j] - pos[i]
        r = float(np.linalg.norm(rij)) + 1e-12
        dr = r - r0
        fij = k * dr / r * rij
        forces[i] += fij
        forces[j] -= fij
        energy += 0.5 * k * dr * dr

    # Repulsão WCA
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in bonded_pairs or (i, j) in one_three:
                continue
            rij = pos[j] - pos[i]
            r = float(np.linalg.norm(rij)) + 1e-12
            sigma, eps = lj_params(symbols[i], symbols[j], radii[i], radii[j])
            r_cut = 2.0**(1.0/6.0) * sigma
            if r < r_cut:
                sr6 = (sigma / r) ** 6
                sr12 = sr6 * sr6
                energy += 4.0 * eps * (sr12 - sr6) + eps
                f_mag = 24.0 * eps / r * (2.0 * sr12 - sr6)
                fij = (f_mag / r) * rij
                forces[i] -= fij
                forces[j] += fij

    return forces, energy