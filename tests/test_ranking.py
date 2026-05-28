"""
test_ranking_validation_bde.py
==============================
Valida se a dinâmica molecular preserva a ordem de estabilidade térmica
baseada em BDEs **experimentais** — referência externa.

Ground truth: BDE experimental (DH°298, kcal/mol)
--------------------------------------------------
Fonte primária : Blanksby & Ellison, Acc. Chem. Res. 2003, 36, 255–263
                 (compilação de experimentos avaliados criticamente)
Fonte secundária: NIST Chemistry WebBook (webbook.nist.gov)
                  CRC Handbook of Chemistry and Physics, 103ª ed.
----------------------------------------------------------

Fontes por ligação (DH°298, kcal/mol)
--------------------------------------
  C–C  (etano)   : 90.1   Blanksby & Ellison (2003) Chart 1
  C–C  (propano) : 88.8   Blanksby & Ellison (2003) Chart 1  [C-CH2-]
  C=C  (etileno) : 174.1  Blanksby & Ellison (2003) Chart 1  [vinylic σ+π]
                   Nota: molécula de etileno — ligação mais fraca é C=C
                   (174 kcal/mol total); sem outra ligação pesada.
  C≡C  (acetileno): 228.0 Blanksby & Ellison (2003)
  N≡N  (N2)      : 225.0  NIST WebBook / Huber & Herzberg 1979
  C–F  (CH3F)    : 115.7  Blanksby & Ellison (2003) Chart 2
  C–Cl (CH3Cl)   : 83.7   Blanksby & Ellison (2003) Chart 2
  C–Br (CH3Br)   : 70.3   Blanksby & Ellison (2003) Chart 2
  C–I  (CH3I)    : 57.1   Blanksby & Ellison (2003) Chart 2
  C=O  (formald.): 180.5  Blanksby & Ellison (2003) — ligação C=O
  C–O  (metanol) : 92.1   Blanksby & Ellison (2003) — CH3–OH
  C–C  (acetald.): 83.0   Blanksby & Ellison (2003) — CH3–CHO (C–C)
                   (mais fraca que C=O nesta molécula)
  C–C  (benzeno) : uso da BDE aromática C–C ≈ 122 kcal/mol (ressonância)
                   Nota: em benzeno todas as ligações pesadas são C–C
                   aromáticas com BDE efetiva ≈ 122 kcal/mol.
                   Fonte: Slayden & Liebman (2001), Chem. Rev.
  C–C  (etanol)  : 90.1   ligação mais fraca é C–C (C–C de etano ~ 90)
                   vs C–O = 92.1  →  C–C é a mais fraca
  C–C  (acetona) : 84.5   CH3–CO (ligação C–C α-carbonílica)
                   Fonte: Blanksby & Ellison (2003)
  C–S  (metanetiol): 74.7 Blanksby & Ellison (2003) — CH3–SH (C–S)

Tabela final (bde_min = BDE da ligação mais fraca entre pesadas)
-----------------------------------------------------------------
  molécula        bde_min (kcal/mol)  ligação mais fraca
  iodometano      57.1               C–I
  bromometano     70.3               C–Br
  metanetiol      74.7               C–S
  clorometano     83.7               C–Cl
  acetaldeido     83.0               C–C (α-carbonílica)
  acetona         84.5               C–C (α-carbonílica)
  propano         88.8               C–C
  etano           90.1               C–C
  etanol          90.1               C–C
  metanol         92.1               C–O
  fluorometano   115.7               C–F
  benzeno        122.0               C–C aromático
  etileno        174.1               C=C
  formaldeido    180.5               C=O
  acetileno      228.0               C≡C
  nitrogenio     225.0               N≡N
  --- excluídos do ranking (só ligações X-H monitoradas pelo motor) ---
  metano           inf               (só C–H)
  amonia           inf               (só N–H)
  agua             inf               (só O–H)
  sulfeto          inf               (só S–H)

Lógica de ranking
-------------------------------------------------
  ranking_difference = n_wrong_pairs / n_comparable_pairs
  pairwise_accuracy  = 1 - ranking_difference

  Um par (fraca, forte) é "correct" se simulated_t(fraca) < simulated_t(forte).
  É "wrong" se a simulação inverteu essa ordem.
"""

from __future__ import annotations

import inspect
import math
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from molecule_lab.simulation import run_simulation
from molecule_lab.simulation.detection import detect_broken_bonds
from molecule_lab.simulation.engine import _RESULT_CACHE
from molecule_lab.simulation.parameters import (
    BOND_ORDER_DEFAULTS,
    BOND_PAIR_PARAMS,
    get_preset,
)
from molecule_lab.simulation.topology import build_molecule, build_topology


# ---------------------------------------------------------------------------
# 20 moléculas em 7 classes químicas
# bde_min  : BDE experimental da ligação mais fraca entre pesadas (kcal/mol)
#            Fonte: Blanksby & Ellison (2003) + NIST WebBook
# weakest_bond_exp : identificação da ligação referenciada
# ---------------------------------------------------------------------------

MOLECULES = [
    # Alcanos — ligação mais fraca é C–C
    {"name": "metano",       "smiles": "C",        "class": "alcano",
     "bde_min": float("inf"), "weakest_bond_exp": "only-CH",
     "bde_note": "sem ligações pesadas — não monitorado pelo motor"},
    {"name": "etano",        "smiles": "CC",       "class": "alcano",
     "bde_min": 90.1, "weakest_bond_exp": "C-C SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 1"},
    {"name": "propano",      "smiles": "CCC",      "class": "alcano",
     "bde_min": 88.8, "weakest_bond_exp": "C-C SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 1 (C-CH2-)"},

    # Halogenados — ordem C-F > C-Cl > C-Br > C-I bem estabelecida
    {"name": "fluorometano", "smiles": "CF",       "class": "halogenado",
     "bde_min": 115.7, "weakest_bond_exp": "C-F SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 2 (CH3F)"},
    {"name": "clorometano",  "smiles": "CCl",      "class": "halogenado",
     "bde_min": 83.7,  "weakest_bond_exp": "C-Cl SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 2 (CH3Cl)"},
    {"name": "bromometano",  "smiles": "CBr",      "class": "halogenado",
     "bde_min": 70.3,  "weakest_bond_exp": "C-Br SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 2 (CH3Br)"},
    {"name": "iodometano",   "smiles": "CI",       "class": "halogenado",
     "bde_min": 57.1,  "weakest_bond_exp": "C-I SINGLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 2 (CH3I)"},

    # Insaturados — ligações múltiplas têm BDE total alta
    {"name": "etileno",      "smiles": "C=C",      "class": "insaturado",
     "bde_min": 174.1, "weakest_bond_exp": "C-C DOUBLE",
     "bde_note": "Blanksby & Ellison 2003, Chart 1 (C=C total)"},
    {"name": "acetileno",    "smiles": "C#C",      "class": "insaturado",
     "bde_min": 228.0, "weakest_bond_exp": "C-C TRIPLE",
     "bde_note": "Blanksby & Ellison 2003 (C≡C total)"},
    {"name": "nitrogenio",   "smiles": "N#N",      "class": "insaturado",
     "bde_min": 225.0, "weakest_bond_exp": "N-N TRIPLE",
     "bde_note": "NIST WebBook / Huber & Herzberg 1979 (N2)"},

    # Heteroátomos simples — apenas ligações X-H, motor não monitora
    {"name": "amonia",       "smiles": "N",        "class": "heteroatomo",
     "bde_min": float("inf"), "weakest_bond_exp": "only-NH",
     "bde_note": "sem ligações pesadas — não monitorado pelo motor"},
    {"name": "agua",         "smiles": "O",        "class": "heteroatomo",
     "bde_min": float("inf"), "weakest_bond_exp": "only-OH",
     "bde_note": "sem ligações pesadas — não monitorado pelo motor"},
    {"name": "sulfeto",      "smiles": "S",        "class": "heteroatomo",
     "bde_min": float("inf"), "weakest_bond_exp": "only-SH",
     "bde_note": "sem ligações pesadas — não monitorado pelo motor"},

    # Carbonílicos
    {"name": "formaldeido",  "smiles": "C=O",      "class": "carbonilico",
     "bde_min": 180.5, "weakest_bond_exp": "C-O DOUBLE",
     "bde_note": "Blanksby & Ellison 2003 (H2C=O total)"},
    {"name": "metanol",      "smiles": "CO",       "class": "carbonilico",
     "bde_min": 92.1,  "weakest_bond_exp": "C-O SINGLE",
     "bde_note": "Blanksby & Ellison 2003 (CH3-OH)"},
    {"name": "acetaldeido",  "smiles": "CC=O",     "class": "carbonilico",
     "bde_min": 83.0,  "weakest_bond_exp": "C-C SINGLE",
     "bde_note": "Blanksby & Ellison 2003 (CH3-CHO, ligação C-C α-carbonílica)"},

    # Aromático — BDE efetiva C–C aromático
    {"name": "benzeno",      "smiles": "c1ccccc1", "class": "aromatico",
     "bde_min": 122.0, "weakest_bond_exp": "C-C AROMATIC",
     "bde_note": "Slayden & Liebman 2001 Chem.Rev. / BDE efetiva ressonância"},

    # Mistos
    {"name": "etanol",       "smiles": "CCO",      "class": "misto",
     "bde_min": 90.1,  "weakest_bond_exp": "C-C SINGLE",
     "bde_note": "ligação mais fraca é C-C (90.1) vs C-O (92.1)"},
    {"name": "acetona",      "smiles": "CC(=O)C",  "class": "misto",
     "bde_min": 84.5,  "weakest_bond_exp": "C-C SINGLE",
     "bde_note": "Blanksby & Ellison 2003 (CH3-CO, α-carbonílica)"},
    {"name": "metanetiol",   "smiles": "CS",       "class": "misto",
     "bde_min": 74.7,  "weakest_bond_exp": "C-S SINGLE",
     "bde_note": "Blanksby & Ellison 2003 (CH3-SH)"},
]


# ---------------------------------------------------------------------------
# Pares esperados — derivados da BDE experimental
# Só incluímos pares com gap > MIN_BDE_GAP (kcal/mol)
# ---------------------------------------------------------------------------

MIN_BDE_GAP = 5.0  # kcal/mol — gap mínimo para o par ser considerado discriminável

def build_expected_order(
    molecules: list[dict],
    min_gap: float = MIN_BDE_GAP,
) -> list[tuple[str, str, float]]:
    """
    Gera pares (name_fraca, name_forte, gap_bde) onde bde_min(fraca) < bde_min(forte)
    e a diferença supera min_gap. Ordena do gap maior para o menor (pares mais
    confiáveis primeiro).
    """
    pairs = []
    for i in range(len(molecules)):
        for j in range(i + 1, len(molecules)):
            a, b = molecules[i], molecules[j]
            if math.isinf(a["bde_min"]) or math.isinf(b["bde_min"]):
                continue  # molécula sem ligações pesadas monitoradas
            gap = abs(a["bde_min"] - b["bde_min"])
            if gap <= min_gap:
                continue
            if a["bde_min"] < b["bde_min"]:
                pairs.append((a["name"], b["name"], gap))
            else:
                pairs.append((b["name"], a["name"], gap))
    pairs.sort(key=lambda x: -x[2])  # maior gap primeiro
    return pairs


EXPECTED_ORDER = build_expected_order(MOLECULES)

# ---------------------------------------------------------------------------
# Lógica de ranking 
# ---------------------------------------------------------------------------

def _numeric(value: Any) -> float:
    return pd.to_numeric(value, errors="coerce")


def compute_pairwise_accuracy(
    results: list[dict],
    expected_order: list[tuple[str, str, float]],
    min_simulated_gap: float = 0.0,
) -> tuple[list[dict], dict]:
    """
    Para cada par (fraca, forte) em expected_order, verifica se
    simulated_t(fraca) < simulated_t(forte).
    """
    by_name = {r["name"]: r for r in results}
    pair_rows: list[dict] = []

    for name_weak, name_strong, bde_gap in expected_order:
        a = by_name.get(name_weak)
        b = by_name.get(name_strong)

        if a is None or b is None:
            pair_rows.append({
                "molecule_weak": name_weak, "molecule_strong": name_strong,
                "bde_gap": bde_gap,
                "simulated_weak": float("nan"), "simulated_strong": float("nan"),
                "reason": "ignored_missing_molecule",
            })
            continue

        sim_w = _numeric(a.get("simulated_t"))
        sim_s = _numeric(b.get("simulated_t"))

        if not a.get("ok", False) or not b.get("ok", False):
            reason = "ignored_failed_simulation"
        elif pd.isna(sim_w) or pd.isna(sim_s):
            reason = "ignored_missing_value"
        elif abs(sim_w - sim_s) <= min_simulated_gap:
            reason = "ignored_simulated_gap"
        elif sim_w < sim_s:
            reason = "correct"
        else:
            reason = "wrong"

        pair_rows.append({
            "molecule_weak":    name_weak,
            "molecule_strong":  name_strong,
            "bde_gap":          bde_gap,
            "simulated_weak":   sim_w,
            "simulated_strong": sim_s,
            "reason":           reason,
        })

    comparable   = [r for r in pair_rows if r["reason"] in ("correct", "wrong")]
    n_correct    = sum(1 for r in comparable if r["reason"] == "correct")
    n_wrong      = len(comparable) - n_correct
    n_comparable = len(comparable)

    metrics = {
        "n_total_pairs":      len(pair_rows),
        "n_comparable_pairs": n_comparable,
        "n_correct_pairs":    n_correct,
        "n_wrong_pairs":      n_wrong,
        "n_ignored_pairs":    len(pair_rows) - n_comparable,
        "pairwise_accuracy":  n_correct / n_comparable if n_comparable else None,
        "ranking_difference": n_wrong   / n_comparable if n_comparable else None,
    }
    return pair_rows, metrics


# ---------------------------------------------------------------------------
# Helpers de simulação
# ---------------------------------------------------------------------------

def _extract_simulated_t(events: list, preset_name: str) -> float:
    NO_BREAK_MARGIN = 1000.0
    preset = get_preset(preset_name)
    t_end  = getattr(preset, "temperature_end", 10_000.0)

    payload = next((e.payload for e in events if e.event == "result"), None)
    if payload is None:
        return float("nan")
    if payload.get("result") == "stable":
        return t_end + NO_BREAK_MARGIN

    break_step   = payload.get("break_step")
    target_temps = payload.get("target_temperatures")
    if break_step is not None and target_temps is not None:
        return float(target_temps[break_step])
    return t_end + NO_BREAK_MARGIN


def _run_all_molecules(preset_name: str, seed: int = 42) -> list[dict]:
    results = []
    for mol in MOLECULES:
        try:
            events = list(run_simulation(mol["smiles"], preset_name=preset_name, seed=seed))
            t = _extract_simulated_t(events, preset_name)
            results.append({
                "name":            mol["name"],
                "smiles":          mol["smiles"],
                "class":           mol["class"],
                "bde_min":         mol["bde_min"],
                "weakest_bond_exp": mol["weakest_bond_exp"],
                "bde_note":        mol["bde_note"],
                "simulated_t":     t,
                "ok":              math.isfinite(t),
            })
        except Exception as exc:
            results.append({
                "name":            mol["name"],
                "smiles":          mol["smiles"],
                "class":           mol["class"],
                "bde_min":         mol["bde_min"],
                "weakest_bond_exp": mol["weakest_bond_exp"],
                "bde_note":        mol["bde_note"],
                "simulated_t":     float("nan"),
                "ok":              False,
                "error":           str(exc),
            })
    return results


# ---------------------------------------------------------------------------
# Testes herdados de test_simulation.py (sem alteração)
# ---------------------------------------------------------------------------

def test_presets_are_available() -> None:
    assert get_preset("fast").n_steps > get_preset("debug").n_steps
    assert get_preset("balanced").event_every > 0


def test_topology_builds_hydrogenated_methane() -> None:
    mol      = build_molecule("C", seed=7)
    topology = build_topology(mol)
    assert len(topology.symbols) == 5
    assert topology.symbols.count("H") == 4
    assert len(topology.bonds) == 4
    assert topology.pos.shape == (5, 3)


def test_detect_broken_bond_controlled_case() -> None:
    pos   = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    bonds = [(0, 1, 1.0, 150.0, 1.0)]

    sig    = inspect.signature(detect_broken_bonds)
    params = sig.parameters
    kwargs: dict = {"break_frac": 0.95}
    if "break_distance_factor" in params:
        default = params["break_distance_factor"].default
        kwargs["break_distance_factor"] = (
            default if default is not inspect.Parameter.empty else 2.0
        )
    broken = detect_broken_bonds(pos, bonds, **kwargs)
    assert len(broken) == 1
    assert broken[0]["bond_index"] == 0


def test_debug_simulation_is_deterministic_and_cached() -> None:
    _RESULT_CACHE.clear()
    first  = list(run_simulation("C", preset_name="debug", seed=123))
    second = list(run_simulation("C", preset_name="debug", seed=123))
    first_result  = next(e.payload for e in first  if e.event == "result")
    second_result = next(e.payload for e in second if e.event == "result")
    assert first_result["result"] == "stable", f"Esperado 'stable', obtido: {first_result}"
    second_events = [e.event for e in second]
    assert "cache_hit" in second_events, (
        f"Esperado evento 'cache_hit'. Eventos: {second_events}"
    )
    assert second_result["result"] == first_result["result"]


# ---------------------------------------------------------------------------
# Testes de validação por ranking — um por preset
# ---------------------------------------------------------------------------

def _test_ranking_for_preset(preset_name: str) -> tuple[list[dict], list[dict], dict]:
    _RESULT_CACHE.clear()
    results = _run_all_molecules(preset_name, seed=42)
    pairs, metrics = compute_pairwise_accuracy(results, EXPECTED_ORDER)
    return results, pairs, metrics


def test_ranking_preserves_experimental_bde_debug() -> None:
    """Preset debug — limiar conservador de 50% dado o preset mínimo."""
    _, _, metrics = _test_ranking_for_preset("debug")
    acc = metrics["pairwise_accuracy"]
    assert acc is not None, "Nenhum par comparável com preset debug."
    assert acc >= 0.50, f"[debug] accuracy={acc:.2%} < 50%"


def test_ranking_preserves_experimental_bde_balanced() -> None:
    """Preset balanced — esperamos pelo menos 65% de acerto."""
    _, _, metrics = _test_ranking_for_preset("balanced")
    acc = metrics["pairwise_accuracy"]
    assert acc is not None, "Nenhum par comparável com preset balanced."
    assert acc >= 0.65, f"[balanced] accuracy={acc:.2%} < 65%"


def test_ranking_preserves_experimental_bde_fast() -> None:
    """Preset fast — rampa mais completa, esperamos pelo menos 75% de acerto."""
    _, _, metrics = _test_ranking_for_preset("fast")
    acc = metrics["pairwise_accuracy"]
    assert acc is not None, "Nenhum par comparável com preset fast."
    assert acc >= 0.75, f"[fast] accuracy={acc:.2%} < 75%"


# ---------------------------------------------------------------------------
# Runner manual
# ---------------------------------------------------------------------------

def _print_ranking_report(
    preset_name: str, results: list[dict], pairs: list[dict], metrics: dict
) -> None:
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"
    C = "\033[96m"; RESET = "\033[0m"; B = "\033[1m"

    acc  = metrics["pairwise_accuracy"]
    diff = metrics["ranking_difference"]
    print(f"\n{B}{C}── Preset: {preset_name.upper()} ──{RESET}")
    print(f"  Moléculas simuladas : {len(results)}")
    print(f"  Pares comparáveis   : {metrics['n_comparable_pairs']} / {metrics['n_total_pairs']}")
    print(f"  Corretos / Errados  : {metrics['n_correct_pairs']} / {metrics['n_wrong_pairs']}")
    print(f"  pairwise_accuracy   : {B}{acc:.2%}{RESET}" if acc is not None else "  pairwise_accuracy   : N/A")
    print(f"  ranking_difference  : {diff:.2%}" if diff is not None else "  ranking_difference  : N/A")

    # Tabela ordenada por BDE experimental e T simulada
    print(f"\n  {'Molécula':<16} {'Classe':<14} {'Lig. fraca (exp)':<20} {'BDE_min':>8}  {'T simul.':>10}  ok")
    print(f"  {'─'*16} {'─'*14} {'─'*20} {'─'*8}  {'─'*10}  {'─'*5}")
    for r in sorted(results, key=lambda x: x.get("bde_min", float("inf"))):
        bde = r["bde_min"]
        bde_str = f"{bde:.1f}" if not math.isinf(bde) else "∞ (só X-H)"
        t   = r.get("simulated_t", float("nan"))
        t_str = f"{t:.0f} K" if math.isfinite(t) else "ERRO"
        ok_str = f"{G}✓{RESET}" if r["ok"] else f"{R}✗{RESET}"
        err = f"  ← {r.get('error','')}" if not r["ok"] else ""
        print(f"  {r['name']:<16} {r['class']:<14} {r['weakest_bond_exp']:<20} {bde_str:>8}  {t_str:>10}  {ok_str}{err}")

    wrong = [p for p in pairs if p["reason"] == "wrong"]
    if wrong:
        print(f"\n  {R}{B}Pares invertidos ({len(wrong)}) — motor diverge da BDE experimental:{RESET}")
        for p in sorted(wrong, key=lambda x: -x["bde_gap"]):
            print(
                f"    {R}✗{RESET}  {p['molecule_weak']:<16} esperado < {p['molecule_strong']:<16}"
                f" | BDE_gap={p['bde_gap']:.1f} kcal/mol"
                f"  |  simulado: {p['simulated_weak']:.0f} K  vs  {p['simulated_strong']:.0f} K"
            )
    else:
        print(f"\n  {G}Nenhum par invertido.{RESET}")

    ignored = [p for p in pairs if p["reason"] not in ("correct", "wrong")]
    if ignored:
        print(f"\n  {Y}Pares ignorados ({len(ignored)}):{RESET}")
        for p in ignored:
            print(f"    {Y}–{RESET}  {p['molecule_weak']:<16} / {p['molecule_strong']:<16}  ({p['reason']})")


if __name__ == "__main__":
    G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; RESET = "\033[0m"; B = "\033[1m"

    # Mostra BDEs experimentais antes de rodar qualquer simulação
    print(f"\n{B}{'='*70}")
    print(" BDE EXPERIMENTAL (Blanksby & Ellison 2003 / NIST WebBook)")
    print(f"{'='*70}{RESET}")
    print(f"  {'Molécula':<16} {'Classe':<14} {'Lig. fraca':<20} {'BDE_min (kcal/mol)':<20} {'Fonte'}")
    print(f"  {'─'*16} {'─'*14} {'─'*20} {'─'*20} {'─'*40}")
    for m in sorted(MOLECULES, key=lambda x: x["bde_min"]):
        bde = m["bde_min"]
        bde_str = f"{bde:.1f}" if not math.isinf(bde) else "∞  (só X-H — excluído)"
        print(f"  {m['name']:<16} {m['class']:<14} {m['weakest_bond_exp']:<20} {bde_str:<20} {m['bde_note']}")
    print(f"\n  Pares comparáveis gerados (gap > {MIN_BDE_GAP} kcal/mol): {len(EXPECTED_ORDER)}")

    base_tests = [
        ("1  presets disponíveis",                test_presets_are_available),
        ("2  topologia do metano",                test_topology_builds_hydrogenated_methane),
        ("3  detecção de ligação quebrada",        test_detect_broken_bond_controlled_case),
        ("4  simulação determinística e cacheada", test_debug_simulation_is_deterministic_and_cached),
    ]
    ranking_presets = [
        ("5  ranking BDE exp — preset debug",    "debug",    0.50),
        ("6  ranking BDE exp — preset balanced", "balanced", 0.65),
        ("7  ranking BDE exp — preset fast",     "fast",     0.75),
    ]

    col_w = 44
    passed, failed = [], []

    print(f"\n{B}{'='*70}")
    print(" TESTES")
    print(f"{'='*70}{RESET}\n")

    for label, fn in base_tests:
        print(f"  {label:<{col_w}}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            fn()
            print(f"{G}PASSOU{RESET}  ({time.perf_counter()-t0:.2f}s)")
            passed.append(label)
        except AssertionError as exc:
            print(f"{R}FALHOU{RESET}  ({time.perf_counter()-t0:.2f}s)\n    {exc}")
            failed.append(label)
        except Exception as exc:
            print(f"{Y}ERRO  {RESET}  ({time.perf_counter()-t0:.2f}s)\n    {exc}")
            failed.append(label)

    print(f"\n{B}{'─'*70}")
    print(" RELATÓRIO DE RANKING POR PRESET  [ground truth: BDE experimental]")
    print(f"{'─'*70}{RESET}")

    for label, preset, threshold in ranking_presets:
        print(f"\n  {label:<{col_w}}", end="", flush=True)
        t0 = time.perf_counter()
        try:
            results, pairs, metrics = _test_ranking_for_preset(preset)
            elapsed = time.perf_counter() - t0
            acc = metrics["pairwise_accuracy"]
            ok  = acc is not None and acc >= threshold
            status = f"{G}PASSOU{RESET}" if ok else f"{R}FALHOU{RESET}"
            acc_str = f"{acc:.2%}" if acc is not None else "N/A"
            print(f"{status}  ({elapsed:.1f}s)  accuracy={acc_str}  (limiar={threshold:.0%})")
            _print_ranking_report(preset, results, pairs, metrics)
            (passed if ok else failed).append(label)
        except Exception as exc:
            print(f"{Y}ERRO  {RESET}  ({time.perf_counter()-t0:.1f}s)\n    {exc}")
            print(traceback.format_exc())
            failed.append(label)

    total = len(base_tests) + len(ranking_presets)
    print(f"\n{B}{'='*70}{RESET}")
    if failed:
        print(f"  {R}{B}RESULTADO: {len(failed)}/{total} falharam{RESET}  ({len(passed)} passaram)")
        print(f"  Falharam: {', '.join(failed)}")
    else:
        print(f"  {G}{B}RESULTADO: todos os {total} testes passaram ✓{RESET}")
    print(f"{B}{'='*70}{RESET}\n")