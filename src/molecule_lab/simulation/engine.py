"""High-level molecular dynamics runner and progress event generation.

This public runner keeps the FastAPI/SSE contract stable while using the
BDE-calibrated MD model added by the validation PR.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from molecule_lab.core.config import settings
from molecule_lab.core.errors import SimulationError
from molecule_lab.simulation.bde_loader import load_bde_table
from molecule_lab.simulation.detection_ import detect_broken_bonds
from molecule_lab.simulation.forces_ import forces_and_energy
from molecule_lab.simulation.integrator_ import (
    initialize_velocities,
    kinetic_temperature,
    target_temperature_ramp,
    velocity_verlet_step,
)
from molecule_lab.simulation.parameters import SimulationPreset, get_preset
from molecule_lab.simulation.parameters_ import (
    NO_BREAK_MARGIN,
    SAVE_EVERY,
    TEMPERATURE_END,
    TEMPERATURE_START,
)
from molecule_lab.simulation.topology_ import build_molecule, build_topology


ResultKind = Literal["stable", "break"]
EventKind = Literal["progress", "result", "cache_hit"]

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BDE_PATH = _PROJECT_ROOT / "data" / "bde_results.csv"
_OPTUNA_PATH = _PROJECT_ROOT / "data" / "optuna_resultado.json"
_MODEL_NAME = "bde_ranking_md_v1"


@dataclass(frozen=True)
class SimulationEvent:
    event: EventKind
    payload: dict[str, Any]


_RESULT_CACHE: OrderedDict[tuple[str, str, int, str], dict[str, Any]] = OrderedDict()


def run_simulation(
    smiles: str, preset_name: str = "fast", seed: int | None = None
) -> Iterator[SimulationEvent]:
    preset = get_preset(preset_name)
    run_seed = preset.seed if seed is None else seed
    runtime = _runtime_params(preset)
    model_signature = _model_signature(runtime)
    cache_key = (smiles, preset.name, run_seed, model_signature)

    cached = _cache_get(cache_key)
    if cached is not None:
        yield SimulationEvent(
            "cache_hit",
            {
                "smiles": smiles,
                "preset": preset.name,
                "seed": run_seed,
                "model": _MODEL_NAME,
                "model_signature": model_signature,
                "message": "Resultado recuperado do cache em memória.",
            },
        )
        yield SimulationEvent("result", {**cached, "cached": True})
        return

    bde_table = _load_bde_table()
    final_payload: dict[str, Any] | None = None
    for event in _run_uncached(smiles, preset, run_seed, runtime, bde_table):
        if event.event == "result":
            final_payload = event.payload
        yield event
    if final_payload is not None:
        _cache_put(cache_key, final_payload)


def _run_uncached(
    smiles: str,
    preset: SimulationPreset,
    seed: int,
    runtime: dict[str, float | int],
    bde_table: dict,
) -> Iterator[SimulationEvent]:
    start = time.perf_counter()
    try:
        mol = build_molecule(smiles, seed=seed)
        pos, masses, radii, symbols, bonds, _neighbors, one_three, bonded_pairs = (
            build_topology(mol)
        )
    except Exception as exc:
        if isinstance(exc, SimulationError):
            raise
        raise SimulationError(str(exc)) from exc

    temperature_start = float(runtime["temperature_start"])
    temperature_end = float(runtime["temperature_end"])
    n_steps = int(runtime["n_steps"])
    dt = float(runtime["dt"])
    thermostat_tau = float(runtime["thermostat_tau"])
    break_factor = float(runtime["break_factor"])
    break_persistence = int(runtime["break_persistence"])
    alpha = float(runtime["alpha"])
    event_every = max(1, int(runtime["event_every"]))

    vel = initialize_velocities(masses, temperature_start, seed=seed)
    forces, potential = forces_and_energy(
        pos, radii, symbols, bonds, one_three, bonded_pairs
    )

    temperatures: list[float] = []
    target_temperatures: list[float] = []
    persist_count = {idx: 0 for idx in range(len(bonds))}
    for step in range(n_steps):
        target_temperature = target_temperature_ramp(
            step,
            n_steps,
            temperature_start,
            temperature_end,
        )

        def force_fn(
            current_pos: np.ndarray,
            current_radii,
            current_symbols,
            current_bonds,
            current_one_three,
            current_bonded_pairs,
        ):
            return forces_and_energy(
                current_pos,
                current_radii,
                current_symbols,
                current_bonds,
                current_one_three,
                current_bonded_pairs,
            )

        pos, vel, forces = velocity_verlet_step(
            pos,
            vel,
            forces,
            masses,
            dt,
            force_fn,
            thermostat_tau,
            target_temperature,
            one_three,
            bonded_pairs,
            radii,
            symbols,
            bonds,
        )

        current_temperature, kinetic = kinetic_temperature(vel, masses)
        temperatures.append(float(current_temperature))
        target_temperatures.append(float(target_temperature))

        broken_now = detect_broken_bonds(
            pos,
            bonds,
            radii,
            symbols,
            break_factor,
            alpha,
            bde_table,
        )
        active = {int(entry["bond_index"]) for entry in broken_now}
        for idx in persist_count:
            persist_count[idx] = persist_count[idx] + 1 if idx in active else 0
        persistent = [
            entry
            for entry in broken_now
            if persist_count[int(entry["bond_index"])] >= break_persistence
        ]

        if step % event_every == 0 or step == n_steps - 1:
            _forces, potential = forces_and_energy(
                pos, radii, symbols, bonds, one_three, bonded_pairs
            )
            yield SimulationEvent(
                "progress",
                {
                    "step": step,
                    "n_steps": n_steps,
                    "progress": round((step + 1) / n_steps, 4),
                    "target_temperature": float(target_temperature),
                    "current_temperature": float(current_temperature),
                    "potential_energy": float(potential),
                    "kinetic_energy": float(kinetic),
                    "candidate_broken_bonds": _format_broken_bonds(
                        broken_now, symbols
                    ),
                    "model": _MODEL_NAME,
                },
            )

        if persistent:
            yield SimulationEvent(
                "result",
                _result_payload(
                    "break",
                    step,
                    float(target_temperature),
                    float(current_temperature),
                    persistent,
                    symbols,
                    temperatures,
                    target_temperatures,
                    start,
                    runtime,
                    preset,
                ),
            )
            return

    yield SimulationEvent(
        "result",
        _result_payload(
            "stable",
            None,
            None,
            None,
            [],
            symbols,
            temperatures,
            target_temperatures,
            start,
            runtime,
            preset,
        ),
    )


def _result_payload(
    result: ResultKind,
    break_step: int | None,
    break_temperature: float | None,
    current_break_temperature: float | None,
    broken_bonds: list[dict[str, float | int | str]],
    symbols: list[str],
    temperatures: list[float],
    target_temperatures: list[float],
    start_time: float,
    runtime: dict[str, float | int],
    preset: SimulationPreset,
) -> dict[str, Any]:
    simulated_break_temperature = (
        break_temperature
        if break_temperature is not None
        else float(runtime["temperature_end"]) + NO_BREAK_MARGIN
    )
    return {
        "result": result,
        "break_step": break_step,
        "break_temperature": break_temperature,
        "simulated_break_temperature": simulated_break_temperature,
        "current_break_temperature": current_break_temperature,
        "broken_bonds": _format_broken_bonds(broken_bonds, symbols),
        "symbols": symbols,
        "temperatures": temperatures,
        "target_temperatures": target_temperatures,
        "elapsed_seconds": round(time.perf_counter() - start_time, 4),
        "cached": False,
        "model": _MODEL_NAME,
        "model_signature": _model_signature(runtime),
        "preset": preset.name,
        "parameters": {
            "temperature_start": float(runtime["temperature_start"]),
            "temperature_end": float(runtime["temperature_end"]),
            "n_steps": int(runtime["n_steps"]),
            "dt": float(runtime["dt"]),
            "thermostat_tau": float(runtime["thermostat_tau"]),
            "break_factor": float(runtime["break_factor"]),
            "break_persistence": int(runtime["break_persistence"]),
            "alpha": float(runtime["alpha"]),
        },
    }


def _format_broken_bonds(
    broken_bonds: list[dict[str, float | int | str]], symbols: list[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for bond in broken_bonds:
        i = int(bond["i"])
        j = int(bond["j"])
        r0 = bond.get("equilibrium_distance", bond.get("r0"))
        output.append(
            {
                "atom_i": symbols[i],
                "atom_j": symbols[j],
                "atom_i_index": i,
                "atom_j_index": j,
                "distance": _rounded_or_none(bond.get("distance")),
                "r0": _rounded_or_none(r0),
                "threshold": _rounded_or_none(bond.get("threshold")),
                "bde": _rounded_or_none(bond.get("bde")),
                "bde_source": str(bond.get("bde_source", "")),
                "bde_factor": _rounded_or_none(bond.get("bde_factor")),
                "reason": str(bond.get("reason", "bde_threshold")),
            }
        )
    return output


def _runtime_params(preset: SimulationPreset) -> dict[str, float | int]:
    params = _load_optuna_params()
    return {
        "temperature_start": TEMPERATURE_START,
        "temperature_end": TEMPERATURE_END,
        "n_steps": int(params["n_steps"]),
        "dt": float(params["dt"]),
        "thermostat_tau": float(params["thermostat_tau"]),
        "break_factor": float(params["break_factor"]),
        "break_persistence": int(params["break_persistence"]),
        "alpha": float(params["alpha"]),
        "save_every": SAVE_EVERY,
        "event_every": max(1, int(preset.event_every)),
    }


def _load_optuna_params() -> dict[str, float | int]:
    try:
        with open(_OPTUNA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        raise SimulationError(f"Arquivo de parâmetros Optuna não encontrado: {exc}") from exc

    params = data.get("best_params", {})
    required = ["dt", "thermostat_tau", "break_factor", "break_persistence", "alpha", "n_steps"]
    missing = [key for key in required if key not in params]
    if missing:
        raise SimulationError(
            f"Parâmetros Optuna ausentes em {_OPTUNA_PATH}: {', '.join(missing)}"
        )
    return params


def _load_bde_table() -> dict:
    try:
        return load_bde_table(_BDE_PATH)
    except Exception as exc:
        raise SimulationError(f"Falha ao carregar tabela BDE: {exc}") from exc


def _model_signature(runtime: dict[str, float | int]) -> str:
    payload = {
        "model": _MODEL_NAME,
        "runtime": runtime,
        "bde": _file_signature(_BDE_PATH),
        "optuna": _file_signature(_OPTUNA_PATH),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _file_signature(path: Path) -> dict[str, float | int | str]:
    try:
        stat = path.stat()
    except OSError:
        return {"path": str(path), "missing": 1}
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _rounded_or_none(value: object, digits: int = 3) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _cache_get(cache_key: tuple[str, str, int, str]) -> dict[str, Any] | None:
    value = _RESULT_CACHE.get(cache_key)
    if value is None:
        return None
    _RESULT_CACHE.move_to_end(cache_key)
    return value


def _cache_put(cache_key: tuple[str, str, int, str], value: dict[str, Any]) -> None:
    _RESULT_CACHE[cache_key] = value
    _RESULT_CACHE.move_to_end(cache_key)
    while len(_RESULT_CACHE) > settings.simulation_cache_size:
        _RESULT_CACHE.popitem(last=False)
