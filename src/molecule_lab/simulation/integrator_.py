#!/usr/bin/env python3
"""Integrador Velocity Verlet + termostato de Berendsen + rampa linear."""

import math
import numpy as np


def initialize_velocities(masses, temperature, seed=123):
    """Velocidades iniciais de Maxwell‑Boltzmann (momentum linear removido)."""
    rng = np.random.default_rng(seed)
    vel = rng.normal(size=(len(masses), 3)) * np.sqrt(0.05 * max(temperature, 1e-6) / masses)[:, None]
    vel -= np.average(vel, axis=0, weights=masses)
    return vel


def kinetic_temperature(vel, masses):
    """Temperatura cinética (DOF = 3N-3)."""
    ke = 0.5 * float(np.sum(masses[:, None] * vel * vel))
    dof = max(1, 3 * len(masses) - 3)
    return 2.0 * ke / dof, ke


def target_temperature_ramp(step, n_steps, t0, t1):
    """Temperatura alvo na rampa linear."""
    if n_steps <= 1:
        return t1
    return t0 + (t1 - t0) * step / (n_steps - 1)


def velocity_verlet_step(pos, vel, forces, masses, dt, force_fn, thermostat_tau,
                         target_t, one_three, bonded_pairs, radii, symbols, bonds):
    """
    Um passo de integração Velocity Verlet com termostato de Berendsen.
    Retorna (pos, vel, novas_forças).
    """
    # Meio passo na velocidade
    vel = vel + 0.5 * dt * (forces / masses[:, None])

    # Posição completa
    pos = pos + dt * vel

    # Novas forças
    new_forces, _ = force_fn(pos, radii, symbols, bonds, one_three, bonded_pairs)

    # Segundo meio passo na velocidade
    vel = vel + 0.5 * dt * (new_forces / masses[:, None])

    # Termostato de Berendsen
    current_t, _ = kinetic_temperature(vel, masses)
    if current_t > 1e-12:
        scale = 1.0 + dt / thermostat_tau * (target_t / current_t - 1.0)
        vel *= math.sqrt(max(0.0, scale))

    return pos, vel, new_forces