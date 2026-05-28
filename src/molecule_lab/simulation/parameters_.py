#!/usr/bin/env python3
"""Parâmetros globais da simulação MD."""

# Temperaturas da rampa (K)
TEMPERATURE_START = 1.0
TEMPERATURE_END   = 10_000.0
NO_BREAK_MARGIN   = 1_000.0   # adicionado quando não há quebra

# Parâmetros Lennard‑Jones (ε em kJ/mol)
EPSILON_MAP = {
    "H":  0.02, "C":  0.08, "N":  0.10, "O":  0.12,
    "S":  0.15, "F":  0.10, "Cl": 0.15, "Br": 0.20, "I":  0.25,
}
SUPPORTED_ATOMS = frozenset(EPSILON_MAP)

# Fator de escala do raio de van der Waals (WCA)
SIGMA_SCALE = 0.85

# Constantes de mola por tipo de ligação (kJ/mol/Å²)
BOND_K = {
    "SINGLE":   300.0,
    "DOUBLE":   500.0,
    "TRIPLE":   700.0,
    "AROMATIC": 450.0,
}

# No final do arquivo, junto com as outras constantes
SAVE_EVERY = 10