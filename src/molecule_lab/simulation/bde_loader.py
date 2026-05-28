#!/usr/bin/env python3
"""Carregamento da tabela de BDE a partir de CSV/JSON."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

_BOND_SYMBOL_TO_TYPE = {"-": "SINGLE", "=": "DOUBLE", "#": "TRIPLE", ":": "AROMATIC"}


def parse_bond_pair(bond_pair_str):
    s = str(bond_pair_str).strip()
    for sym, bt in _BOND_SYMBOL_TO_TYPE.items():
        if sym in s:
            parts = s.split(sym, 1)
            a, b = parts[0].strip().capitalize(), parts[1].strip().capitalize()
            return min(a, b), max(a, b), bt
    parts = s.split()
    if len(parts) == 2:
        a, b = parts[0].capitalize(), parts[1].capitalize()
        return min(a, b), max(a, b), "SINGLE"
    raise ValueError(f"Não foi possível parsear bond_pair: {bond_pair_str!r}")


def load_bde_table(bde_data_path: str | Path) -> dict:
    path = Path(bde_data_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de BDE não encontrado: {bde_data_path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            rows = raw
        else:
            rows = [
                {"bond_pair": k, "BDE_kJ_mol": v, "status": "ok"}
                for k, v in raw.items()
            ]
    else:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    required = {"bond_pair", "BDE_kJ_mol"}
    columns = set().union(*(row.keys() for row in rows)) if rows else set()
    if not required.issubset(columns):
        raise ValueError(f"Colunas obrigatórias faltando: {required}")

    table = {}
    for row in rows:
        if str(row.get("status", "ok")).lower() != "ok":
            continue
        bde = _to_float(row.get("BDE_kJ_mol"))
        if bde is None:
            continue
        try:
            a, b, bt = parse_bond_pair(row["bond_pair"])
        except ValueError as e:
            print(f"Aviso: ignorando linha - {e}")
            continue
        key = (a, b, bt)
        table[key] = (table[key] + bde) / 2.0 if key in table else bde

    if not table:
        raise ValueError("Nenhuma BDE válida carregada.")

    ch_key = ("C", "H", "SINGLE")
    bde_ref = table.get(ch_key, min(table.values()))
    if ch_key not in table:
        print(f"Aviso: C-H SINGLE não encontrado. Usando menor BDE como REF: {bde_ref:.1f}")

    min_bde = min(table.values())
    table["REF"] = bde_ref
    table["_fallback_ch"] = table.get(ch_key, min_bde)
    table["_min_bde"] = min_bde
    return table


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
