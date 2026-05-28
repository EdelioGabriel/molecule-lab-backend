#!/usr/bin/env python3
"""Detecção de quebra de ligações com persistência."""

import numpy as np


def get_bde_with_fallback(sym_i, sym_j, bond_type, bde_table):
    """Retorna (BDE, fonte) usando fallback hierárquico."""
    key = (min(sym_i, sym_j), max(sym_i, sym_j), bond_type)
    if key in bde_table and isinstance(bde_table[key], float):
        return bde_table[key], "dados_exato"

    # Fallback: SINGLE do mesmo par
    key_single = (min(sym_i, sym_j), max(sym_i, sym_j), "SINGLE")
    if key_single in bde_table and isinstance(bde_table[key_single], float):
        return bde_table[key_single], f"fallback_SINGLE_{min(sym_i,sym_j)}-{max(sym_i,sym_j)}"

    # Fallback: C-H SINGLE (ou menor BDE)
    ch_bde = bde_table.get("_fallback_ch", bde_table.get("_min_bde"))
    if ch_bde is not None:
        return float(ch_bde), "fallback_C-H_SINGLE"

    min_bde = bde_table.get("_min_bde", 300.0)
    return float(min_bde), "fallback_min_dataset"


def detect_broken_bonds(pos, bonds, radii, symbols, factor, alpha, bde_table):
    """
    Detecta ligações pesadas (não‑H) cujo comprimento excede:
        threshold = factor * r0 * (BDE / BDE_REF)^alpha
    """
    bde_ref = float(bde_table["REF"])
    broken = []
    for idx, (i, j, r0, _k, bond_type) in enumerate(bonds):
        if symbols[i] == "H" or symbols[j] == "H":
            continue
        dist = float(np.linalg.norm(pos[i] - pos[j]))
        bde, source = get_bde_with_fallback(symbols[i], symbols[j], bond_type, bde_table)
        bde_factor = (bde / bde_ref) ** alpha
        threshold = factor * r0 * bde_factor
        if dist > threshold:
            broken.append({
                "bond_index": idx,
                "i": i, "j": j,
                "distance": dist,
                "threshold": threshold,
                "equilibrium_distance": r0,
                "bde": bde,
                "bde_source": source,
                "bde_factor": bde_factor,
            })
    return broken