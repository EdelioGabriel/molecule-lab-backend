#!/usr/bin/env python3
"""Motor MD de alto nível: executa a simulação com rampa de temperatura."""

import numpy as np

from .parameters_ import TEMPERATURE_START, TEMPERATURE_END, SAVE_EVERY
from .topology_ import build_molecule, build_topology, set_positions
from .forces_ import forces_and_energy
from .integrator_ import (initialize_velocities, kinetic_temperature,
                        target_temperature_ramp, velocity_verlet_step)
from .detection_ import detect_broken_bonds


def run_md(smiles, bde_table,
           temperature_start=TEMPERATURE_START,
           temperature_end=TEMPERATURE_END,
           n_steps=12000,
           dt=2.5e-4,
           thermostat_tau=0.03,
           break_factor=2.0,
           break_persistence=20,
           alpha=1.0,
           save_every=SAVE_EVERY):
    """
    Executa a dinâmica molecular com rampa linear de temperatura.

    Retorna um dicionário com:
        - mol, trajectory, saved_steps
        - temperatures, target_temperatures
        - bonds, symbols
        - persistent_broken, break_step
    """
    mol = build_molecule(smiles)
    pos, masses, radii, symbols, bonds, _, one_three, bonded_pairs = build_topology(mol)

    vel = initialize_velocities(masses, temperature_start)
    forces, _ = forces_and_energy(pos, radii, symbols, bonds, one_three, bonded_pairs)

    trajectory = [pos.copy()]
    temperatures = []
    target_temps = []
    saved_steps = [0]
    persist = {idx: 0 for idx in range(len(bonds))}

    for step in range(n_steps):
        target_t = target_temperature_ramp(step, n_steps, temperature_start, temperature_end)

        # Integração
        def force_fn(p, r, s, b, ot, bp):
            return forces_and_energy(p, r, s, b, ot, bp)

        pos, vel, forces = velocity_verlet_step(
            pos, vel, forces, masses, dt, force_fn, thermostat_tau, target_t,
            one_three, bonded_pairs, radii, symbols, bonds
        )

        # Registro
        current_t, _ = kinetic_temperature(vel, masses)
        temperatures.append(current_t)
        target_temps.append(target_t)

        # Detecção de quebra
        currently_broken = detect_broken_bonds(
            pos, bonds, radii, symbols, break_factor, alpha, bde_table
        )
        active = {e["bond_index"] for e in currently_broken}
        for idx in persist:
            persist[idx] = persist[idx] + 1 if idx in active else 0

        persistent = [e for e in currently_broken
                      if persist[e["bond_index"]] >= break_persistence]

        if (step + 1) % save_every == 0:
            trajectory.append(pos.copy())
            saved_steps.append(step + 1)

        if persistent:
            set_positions(mol, pos)
            if saved_steps[-1] != step + 1:
                trajectory.append(pos.copy())
                saved_steps.append(step + 1)
            return {
                "mol": mol,
                "trajectory": trajectory,
                "saved_steps": np.array(saved_steps),
                "temperatures": np.array(temperatures),
                "target_temperatures": np.array(target_temps),
                "bonds": bonds,
                "symbols": symbols,
                "persistent_broken": persistent,
                "break_step": step,
            }

    set_positions(mol, pos)
    if saved_steps[-1] != n_steps:
        trajectory.append(pos.copy())
        saved_steps.append(n_steps)
    return {
        "mol": mol,
        "trajectory": trajectory,
        "saved_steps": np.array(saved_steps),
        "temperatures": np.array(temperatures),
        "target_temperatures": np.array(target_temps),
        "bonds": bonds,
        "symbols": symbols,
        "persistent_broken": [],
        "break_step": None,
    }