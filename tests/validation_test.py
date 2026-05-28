#!/usr/bin/env python3
"""
run_validation.py
=================
Validação científica por ranking para MD com rampa de temperatura.
Utiliza os módulos de simulação localizados em src/molecule_lab/simulation.

Uso:
    python run_validation.py --input ../data/validation_molecules.csv --bde ../data/bde_results.csv --output ../results
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Adiciona o diretório src/ ao path para importar molecule_lab.simulation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Imports dos módulos da simulação
from molecule_lab.simulation.engine_ import run_md
from molecule_lab.simulation.bde_loader import load_bde_table
from molecule_lab.simulation.parameters_ import TEMPERATURE_START, TEMPERATURE_END, NO_BREAK_MARGIN
from molecule_lab.simulation.topology_ import build_molecule  # necessário para get_molecular_formula

# =============================================================================
# Funções auxiliares (caso não existam nos módulos)
# =============================================================================

def get_molecular_formula(smiles: str) -> str:
    """Retorna fórmula molecular a partir do SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    return rdMolDescriptors.CalcMolFormula(mol) if mol is not None else ""

def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if not (math.isnan(value) or math.isinf(value)) else None
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

def _to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "sim", "y"}

def _normalize_unit(unit) -> str:
    if pd.isna(unit):
        return ""
    compact = (str(unit).strip().lower()
               .replace(" ", "").replace(".", "")
               .replace("-", "").replace("_", ""))
    if compact in {"k", "kelvin"}:
        return "K"
    if compact in {"c", "°c", "celsius"}:
        return "C"
    if compact in {"kj/mol", "kjmol"}:
        return "kJ/mol"
    if compact in {"kcal/mol", "kcalmol"}:
        return "kcal/mol"
    return str(unit).strip()

def _apply_reference_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["reference_value_original"] = pd.to_numeric(df["reference_value"], errors="coerce")
    df["reference_unit_original"] = df.get("reference_unit", "").fillna("")
    rows = []
    for _, row in df.iterrows():
        unit = _normalize_unit(row.get("reference_unit", ""))
        value = pd.to_numeric(row.get("reference_value"), errors="coerce")
        direct_requested = _to_bool(row.get("direct_error_allowed", False))
        norm_value, norm_unit = value, unit
        if pd.notna(value):
            if unit == "C":
                norm_value, norm_unit = float(value) + 273.15, "K"
            elif unit == "kcal/mol":
                norm_value, norm_unit = float(value) * 4.184, "kJ/mol"
        ranking_allowed = pd.notna(value)
        direct_error_allowed_final = direct_requested and norm_unit == "K"
        ref_type = str(row.get("reference_type", "")) if pd.notna(row.get("reference_type")) else ""
        rows.append({
            "reference_strategy": ref_type or "direct_csv",
            "ranking_allowed": ranking_allowed,
            "direct_error_allowed_final": direct_error_allowed_final,
            "reference_direction": "higher_is_more_stable",
            "confidence": "direct",
            "strategy_reason": f"CSV direto: reference_type='{ref_type}', unit='{unit}'",
            "strategy_warnings": "",
            "reference_value_normalized": norm_value,
            "reference_unit_normalized": norm_unit,
        })
    strategy_df = pd.DataFrame(rows)
    return pd.concat([df.reset_index(drop=True), strategy_df.reset_index(drop=True)], axis=1)

def load_validation_data(input_path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"name", "smiles", "reference_value", "reference_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV sem colunas obrigatórias: {sorted(missing)}")
    defaults = {"reference_unit": "", "direct_error_allowed": False, "source": "", "notes": ""}
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df["direct_error_allowed"] = df["direct_error_allowed"].apply(_to_bool)
    df["reference_value"] = pd.to_numeric(df["reference_value"], errors="coerce")
    return _apply_reference_strategy(df)

def load_optuna_params(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    params = data["best_params"]
    required = ["dt", "thermostat_tau", "break_factor", "break_persistence", "alpha", "n_steps"]
    for key in required:
        if key not in params:
            raise ValueError(f"Parâmetro ausente em {path}: {key}")
    return params

def simulate_molecule(row: dict, params: dict, bde_table: dict) -> dict:
    base = {
        "name": row.get("name", ""),
        "formula": get_molecular_formula(row.get("smiles", "")),
        "smiles": row.get("smiles", ""),
        "reference_value": row.get("reference_value", float("nan")),
        "reference_unit": row.get("reference_unit", ""),
        "reference_type": row.get("reference_type", ""),
        "direct_error_allowed": bool(row.get("direct_error_allowed", False)),
        "source": row.get("source", ""),
        "notes": row.get("notes", ""),
        "reference_strategy": row.get("reference_strategy", ""),
        "ranking_allowed": bool(row.get("ranking_allowed", False)),
        "direct_error_allowed_final": bool(row.get("direct_error_allowed_final", False)),
        "reference_value_normalized": row.get("reference_value_normalized", float("nan")),
        "reference_unit_normalized": row.get("reference_unit_normalized", ""),
        "strategy_confidence": row.get("confidence", ""),
        "strategy_reason": row.get("strategy_reason", ""),
        "strategy_warnings": row.get("strategy_warnings", ""),
    }
    try:
        result = run_md(
            smiles=row["smiles"],
            bde_table=bde_table,
            temperature_start=TEMPERATURE_START,
            temperature_end=TEMPERATURE_END,
            n_steps=int(params["n_steps"]),
            dt=float(params["dt"]),
            thermostat_tau=float(params["thermostat_tau"]),
            break_factor=float(params["break_factor"]),
            break_persistence=int(params["break_persistence"]),
            alpha=float(params["alpha"]),
            save_every=10,
        )
        broke = bool(result["persistent_broken"])
        first = result["persistent_broken"][0] if broke else {}
        if broke:
            break_step = int(result["break_step"])
            simulated_t = float(result["target_temperatures"][break_step])
            broken_bond = f"{first['i']}-{first['j']} ({result['symbols'][first['i']]}-{result['symbols'][first['j']]})"
        else:
            simulated_t = float(TEMPERATURE_END + NO_BREAK_MARGIN)
            broken_bond = ""
        return {
            **base,
            "simulated_break_temperature": simulated_t,
            "broke": broke,
            "broken_bond": broken_bond,
            "bde": first.get("bde", float("nan")),
            "bde_source": first.get("bde_source", ""),
            "distance": first.get("distance", float("nan")),
            "threshold": first.get("threshold", float("nan")),
            "error_status": "ok",
            "error_notes": "Simulação concluída",
        }
    except Exception as exc:
        return {
            **base,
            "simulated_break_temperature": float("nan"),
            "broke": False,
            "broken_bond": "",
            "bde": float("nan"),
            "bde_source": "",
            "distance": float("nan"),
            "threshold": float("nan"),
            "error_status": "failed",
            "error_notes": str(exc),
        }

def run_all_simulations(validation_df, params, bde_table):
    rows = []
    for _, row in validation_df.iterrows():
        print(f"Simulando: {row['name']} ({row['smiles']})")
        rows.append(simulate_molecule(row.to_dict(), params, bde_table))
    results_df = pd.DataFrame(rows)
    results_df, error_metrics = compute_error_metrics(results_df)
    results_df = add_rank_classification(results_df)
    return results_df, error_metrics

def compute_error_metrics(results_df):
    df = results_df.copy()
    for col in ("absolute_error", "absolute_error_abs", "percent_error", "absolute_percent_error"):
        df[col] = float("nan")
    for idx, row in df.iterrows():
        can_compare = (bool(row.get("direct_error_allowed_final", False))
                       and row["error_status"] == "ok"
                       and row.get("reference_unit_normalized", "") == "K"
                       and pd.notna(row.get("reference_value_normalized"))
                       and row["reference_value_normalized"] != 0
                       and pd.notna(row["simulated_break_temperature"]))
        if can_compare:
            diff = row["simulated_break_temperature"] - row["reference_value_normalized"]
            df.at[idx, "absolute_error"] = diff
            df.at[idx, "absolute_error_abs"] = abs(diff)
            df.at[idx, "percent_error"] = 100.0 * diff / row["reference_value_normalized"]
            df.at[idx, "absolute_percent_error"] = 100.0 * abs(diff) / row["reference_value_normalized"]
    valid_abs = df["absolute_error_abs"].dropna()
    valid_pct = df["absolute_percent_error"].dropna()
    valid_signed = df.loc[df["absolute_percent_error"].notna(), "percent_error"].astype(float)
    if len(valid_pct) == 0:
        metrics = {"n_error_valid": 0, "mean_absolute_error": None, "median_absolute_error": None,
                   "root_mean_squared_error": None, "mean_percent_error": None,
                   "mean_absolute_percent_error": None, "median_absolute_percent_error": None,
                   "max_absolute_percent_error": None, "min_absolute_percent_error": None}
    else:
        diffs = df.loc[df["absolute_percent_error"].notna(), "absolute_error"].astype(float)
        metrics = {
            "n_error_valid": int(len(valid_pct)),
            "mean_absolute_error": float(valid_abs.mean()),
            "median_absolute_error": float(valid_abs.median()),
            "root_mean_squared_error": float(np.sqrt(np.mean(np.square(diffs)))),
            "mean_percent_error": float(valid_signed.mean()),
            "mean_absolute_percent_error": float(valid_pct.mean()),
            "median_absolute_percent_error": float(valid_pct.median()),
            "max_absolute_percent_error": float(valid_pct.max()),
            "min_absolute_percent_error": float(valid_pct.min()),
        }
    return df, metrics

def add_rank_classification(results_df):
    df = results_df.copy()
    ok = (df["error_status"] == "ok") & df["ranking_allowed"].astype(bool) & df["reference_value_normalized"].notna() & df["simulated_break_temperature"].notna()
    for col in ("reference_rank", "simulated_rank", "rank_delta"):
        df[col] = float("nan")
    for col in ("reference_class", "simulated_class"):
        df[col] = ""
    df["class_correct"] = pd.Series([pd.NA] * len(df), dtype="object")
    df["rank_position_correct"] = pd.Series([pd.NA] * len(df), dtype="object")
    if ok.sum() == 0:
        return df
    df.loc[ok, "reference_rank"] = df.loc[ok, "reference_value_normalized"].rank(method="min", ascending=False).astype(int)
    df.loc[ok, "simulated_rank"] = df.loc[ok, "simulated_break_temperature"].rank(method="min", ascending=False).astype(int)
    df.loc[ok, "rank_delta"] = df.loc[ok, "simulated_rank"] - df.loc[ok, "reference_rank"]
    n_ok = int(ok.sum())
    class_size = max(1, math.ceil(n_ok / 3))
    def _class(rank):
        if pd.isna(rank):
            return ""
        r = int(rank)
        if r <= class_size:
            return "alta_estabilidade"
        if r <= 2 * class_size:
            return "media_estabilidade"
        return "baixa_estabilidade"
    df.loc[ok, "reference_class"] = df.loc[ok, "reference_rank"].apply(_class)
    df.loc[ok, "simulated_class"] = df.loc[ok, "simulated_rank"].apply(_class)
    df.loc[ok, "class_correct"] = df.loc[ok, "reference_class"] == df.loc[ok, "simulated_class"]
    df.loc[ok, "rank_position_correct"] = df.loc[ok, "rank_delta"].abs() <= 2
    return df

def compute_pairwise_accuracy(results_df, min_reference_gap=0.0, min_simulated_gap=0.0):
    rows = []
    n = len(results_df)
    total_pairs = 0
    for a_idx in range(n):
        for b_idx in range(a_idx+1, n):
            total_pairs += 1
            a, b = results_df.iloc[a_idx], results_df.iloc[b_idx]
            ref_a = pd.to_numeric(a.get("reference_value_normalized"), errors="coerce")
            ref_b = pd.to_numeric(b.get("reference_value_normalized"), errors="coerce")
            sim_a = pd.to_numeric(a["simulated_break_temperature"], errors="coerce")
            sim_b = pd.to_numeric(b["simulated_break_temperature"], errors="coerce")
            def order(x, y, gap):
                if pd.isna(x) or pd.isna(y):
                    return "ignored"
                if abs(x - y) <= gap:
                    return "tie"
                return "A_above_B" if x > y else "A_below_B"
            ref_order = order(ref_a, ref_b, min_reference_gap)
            sim_order = order(sim_a, sim_b, min_simulated_gap)
            if a["error_status"] != "ok" or b["error_status"] != "ok":
                reason = "ignored_failed_simulation"
            elif not bool(a.get("ranking_allowed")) or not bool(b.get("ranking_allowed")):
                reason = "ignored_missing_value"
            elif ref_order == "ignored" or sim_order == "ignored":
                reason = "ignored_missing_value"
            elif ref_order == "tie" or sim_order == "tie":
                reason = "ignored_gap"
            elif ref_order == sim_order:
                reason = "correct"
            else:
                reason = "wrong"
            rows.append({
                "molecule_a": a["name"], "molecule_b": b["name"],
                "reference_a": ref_a, "reference_b": ref_b,
                "simulated_a": sim_a, "simulated_b": sim_b,
                "reference_order": ref_order, "simulated_order": sim_order,
                "correct": (reason == "correct") if reason in {"correct","wrong"} else float("nan"),
                "reason": reason,
            })
    pairwise_df = pd.DataFrame(rows)
    comparable = pairwise_df[pairwise_df["reason"].isin(["correct","wrong"])]
    n_correct = int((comparable["reason"] == "correct").sum())
    n_wrong = int((comparable["reason"] == "wrong").sum())
    n_comparable = int(len(comparable))
    metrics = {
        "n_total_pairs": total_pairs,
        "n_comparable_pairs": n_comparable,
        "n_correct_pairs": n_correct,
        "n_wrong_pairs": n_wrong,
        "n_ignored_pairs": total_pairs - n_comparable,
        "pairwise_accuracy": float(n_correct / n_comparable) if n_comparable else None,
    }
    return pairwise_df, metrics

def compute_rank_correlations(results_df):
    valid = results_df[(results_df["error_status"] == "ok") & results_df["ranking_allowed"].astype(bool) & results_df["reference_value_normalized"].notna() & results_df["simulated_break_temperature"].notna()]
    null_result = {"spearman_correlation": None, "spearman_pvalue": None, "kendall_tau": None, "kendall_pvalue": None}
    if len(valid) < 2:
        return null_result
    try:
        from scipy.stats import kendalltau, spearmanr
    except ImportError:
        print("Aviso: scipy não instalado; correlações omitidas.")
        return null_result
    sp = spearmanr(valid["reference_value_normalized"], valid["simulated_break_temperature"])
    kt = kendalltau(valid["reference_value_normalized"], valid["simulated_break_temperature"])
    return {
        "spearman_correlation": float(sp.statistic) if pd.notna(sp.statistic) else None,
        "spearman_pvalue": float(sp.pvalue) if pd.notna(sp.pvalue) else None,
        "kendall_tau": float(kt.statistic) if pd.notna(kt.statistic) else None,
        "kendall_pvalue": float(kt.pvalue) if pd.notna(kt.pvalue) else None,
    }

def build_league_tables(results_df):
    valid = results_df[(results_df["error_status"] == "ok") & results_df["ranking_allowed"].astype(bool) & results_df["reference_value_normalized"].notna() & results_df["simulated_break_temperature"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    real = valid.sort_values(["reference_value_normalized", "name"], ascending=[False, True]).reset_index(drop=True)
    simulated = valid.sort_values(["simulated_break_temperature", "name"], ascending=[False, True]).reset_index(drop=True)
    pos_by_real = {name: i+1 for i, name in enumerate(real["name"])}
    pos_by_sim = {name: i+1 for i, name in enumerate(simulated["name"])}
    valid["reference_position"] = valid["name"].map(pos_by_real)
    valid["simulated_position"] = valid["name"].map(pos_by_sim)
    valid["rank_difference"] = valid["simulated_position"] - valid["reference_position"]
    valid["abs_rank_difference"] = valid["rank_difference"].abs()
    keep_ref = ["reference_position", "simulated_position", "rank_difference", "name", "smiles", "formula", "reference_value_normalized", "simulated_break_temperature", "reference_strategy", "broke", "broken_bond"]
    keep_sim = ["simulated_position", "reference_position", "rank_difference", "name", "smiles", "formula", "simulated_break_temperature", "reference_value_normalized", "reference_strategy", "broke", "broken_bond"]
    molecule_league = valid.sort_values("reference_position")[keep_ref].copy()
    simulated_league = valid.sort_values("simulated_position")[keep_sim].copy()
    league_comparison = valid.sort_values("reference_position")[["name", "reference_position", "simulated_position", "rank_difference", "abs_rank_difference", "reference_value_normalized", "simulated_break_temperature"]].copy()
    league_comparison["status"] = league_comparison["abs_rank_difference"].apply(lambda d: "same" if d==0 else "near" if d<=2 else "moderate" if d<=5 else "large")
    side_by_side = pd.DataFrame([{
        "position": i+1,
        "real_compound": real.iloc[i]["name"],
        "real_formula": real.iloc[i].get("formula", ""),
        "reference_value": real.iloc[i]["reference_value_normalized"],
        "simulated_compound": simulated.iloc[i]["name"],
        "simulated_formula": simulated.iloc[i].get("formula", ""),
        "simulated_temperature": simulated.iloc[i]["simulated_break_temperature"],
        "position_match": real.iloc[i]["name"] == simulated.iloc[i]["name"],
    } for i in range(len(valid))])
    def overlap(k, top):
        k = min(k, len(valid))
        if k == 0:
            return None
        ref_set = set((molecule_league.head(k) if top else molecule_league.tail(k))["name"])
        sim_set = set((simulated_league.head(k) if top else simulated_league.tail(k))["name"])
        return len(ref_set & sim_set) / k
    league_metrics = {
        "n_ranked_molecules": len(valid),
        "mean_abs_rank_difference": float(league_comparison["abs_rank_difference"].mean()),
        "median_abs_rank_difference": float(league_comparison["abs_rank_difference"].median()),
        "max_abs_rank_difference": int(league_comparison["abs_rank_difference"].max()),
        "top_3_overlap": overlap(3, top=True),
        "top_5_overlap": overlap(5, top=True),
        "bottom_3_overlap": overlap(3, top=False),
        "bottom_5_overlap": overlap(5, top=False),
        "exact_position_matches": int((league_comparison["abs_rank_difference"] == 0).sum()),
        "near_position_matches_le2": int((league_comparison["abs_rank_difference"] <= 2).sum()),
    }
    return side_by_side, molecule_league, simulated_league, league_comparison, league_metrics

# =============================================================================
# Gráficos
# =============================================================================

def plot_league_table(side_by_side, output_path):
    if output_path is None or len(side_by_side) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, max(7, len(side_by_side)*0.34)))
    ax.axis("off")
    display = side_by_side.copy()
    display["reference_value"] = display["reference_value"].map(lambda v: f"{v:.0f}")
    display["simulated_temperature"] = display["simulated_temperature"].map(lambda v: f"{v:.0f}")
    display["match"] = display["position_match"].map(lambda v: "OK" if v else "")
    table_data = display[["position", "real_compound", "reference_value", "simulated_compound", "simulated_temperature", "match"]].values
    table = ax.table(cellText=table_data,
                     colLabels=["#", "Referência", "Valor ref.", "Simulado", "T sim.", "Acerto"],
                     loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.25)
    for (row,col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif col == 5:
            cell.set_facecolor("#c7e9c0" if display.iloc[row-1]["position_match"] else "#fdd0a2")
    ax.set_title("Ranking: referência vs simulação", pad=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_league_position_comparison(league_comparison, output_path):
    if output_path is None or len(league_comparison) == 0:
        return
    data = league_comparison.sort_values("reference_position")
    fig, ax = plt.subplots(figsize=(9, max(6, len(data)*0.28)))
    for _, row in data.iterrows():
        color = "#2ca25f" if row["abs_rank_difference"] <= 2 else "#de2d26"
        ax.plot([0,1], [row["reference_position"], row["simulated_position"]], color=color, alpha=0.75)
        ax.text(-0.03, row["reference_position"], row["name"], ha="right", va="center", fontsize=8)
        ax.text(1.03, row["simulated_position"], row["name"], ha="left", va="center", fontsize=8)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Referência", "Simulação"])
    ax.set_ylabel("Posição no ranking"); ax.invert_yaxis()
    ax.set_title("Comparação de posições: referência vs simulação")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_rank_difference(league_comparison, output_path):
    if output_path is None or len(league_comparison) == 0:
        return
    data = league_comparison.sort_values("rank_difference")
    colors = ["#2b8cbe" if v<0 else "#e34a33" if v>0 else "#2ca25f" for v in data["rank_difference"]]
    fig, ax = plt.subplots(figsize=(max(9, len(data)*0.38), 5))
    ax.bar(data["name"], data["rank_difference"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.tick_params(axis="x", rotation=60, labelsize=8)
    ax.set_ylabel("Diferença de posição (simulada − referência)")
    ax.set_title("Diferença de posição no ranking por molécula")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_scatter_ref_vs_sim(results_df, output_path):
    valid = results_df[(results_df["error_status"] == "ok") & results_df["ranking_allowed"].astype(bool) & results_df["reference_value_normalized"].notna()]
    if output_path is None or len(valid) == 0:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(valid["reference_value_normalized"], valid["simulated_break_temperature"], s=55)
    if len(valid) <= 30:
        for _, row in valid.iterrows():
            ax.annotate(row["name"], (row["reference_value_normalized"], row["simulated_break_temperature"]), fontsize=7)
    ax.set_xlabel("Valor de referência")
    ax.set_ylabel("Temperatura simulada de quebra (K)")
    ax.set_title("Referência vs temperatura simulada")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_pairwise_matrix(results_df, pairwise_df, output_path):
    if output_path is None or len(results_df) == 0:
        return
    names = list(results_df["name"])
    lookup = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=int)
    for _, row in pairwise_df.iterrows():
        i = lookup[row["molecule_a"]]; j = lookup[row["molecule_b"]]
        v = 1 if row["reason"] == "correct" else 2 if row["reason"] == "wrong" else 3
        matrix[i,j] = matrix[j,i] = v
    fig, ax = plt.subplots(figsize=(max(6, len(names)*0.7), max(5, len(names)*0.6)))
    cmap = ListedColormap(["#f4f4f4", "#2ca25f", "#de2d26", "#cfcfcf"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Molécula B"); ax.set_ylabel("Molécula A")
    ax.set_title("Acertos par-a-par")
    ax.legend(handles=[Patch(color="#2ca25f", label="ordem preservada"),
                       Patch(color="#de2d26", label="ordem invertida"),
                       Patch(color="#cfcfcf", label="ignorado")],
              loc="upper left", bbox_to_anchor=(1.02,1.0), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_confusion_matrix(pairwise_df, output_path, normalized=False):
    if output_path is None:
        return
    labels = ["A_above_B", "A_below_B"]
    raw = np.zeros((2,2), dtype=float)
    for _, row in pairwise_df[pairwise_df["reason"].isin(["correct","wrong"])].iterrows():
        if row["reference_order"] in labels and row["simulated_order"] in labels:
            raw[labels.index(row["reference_order"]), labels.index(row["simulated_order"])] += 1
    if normalized:
        row_sums = raw.sum(axis=1, keepdims=True)
        matrix = np.divide(raw, row_sums, out=np.zeros_like(raw), where=row_sums!=0)
    else:
        matrix = raw
    fig, ax = plt.subplots(figsize=(7,5.6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusão normalizada" if normalized else "Matriz de confusão")
    ax.set_xticks([0,1]); ax.set_xticklabels(["A acima de B", "A abaixo de B"], rotation=15, ha="right")
    ax.set_yticks([0,1]); ax.set_yticklabels(["A acima de B", "A abaixo de B"])
    ax.set_xlabel("Ordem simulada"); ax.set_ylabel("Ordem de referência")
    for i in range(2):
        for j in range(2):
            if normalized:
                text = f"{100*matrix[i,j]:.1f}%"
            else:
                row_sum = raw[i].sum()
                pct = 0.0 if row_sum==0 else 100.0*raw[i,j]/row_sum
                text = f"{int(raw[i,j])}\n({pct:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_dashboard(results_df, pairwise_df, league_data, output_path):
    side_by_side, _, _, league_comparison, _ = league_data
    fig, axes = plt.subplots(2,2, figsize=(15,11))
    fig.suptitle("Validação do modelo MD — resumo", fontsize=16)
    # Plot 1: tabela lado a lado (top 10)
    ax = axes[0,0]
    ax.axis("off")
    display = side_by_side.head(10).copy()
    if len(display) > 0:
        display["reference_value"] = display["reference_value"].map(lambda v: f"{v:.0f}")
        display["simulated_temperature"] = display["simulated_temperature"].map(lambda v: f"{v:.0f}")
        display["match"] = display["position_match"].map(lambda v: "✓" if v else "✗")
        table_data = display[["position", "real_compound", "reference_value", "simulated_compound", "simulated_temperature", "match"]].values
        table = ax.table(cellText=table_data,
                         colLabels=["#", "Referência", "Valor", "Simulado", "T sim.", "Acerto"],
                         loc="center", cellLoc="left", colLoc="left", fontsize=6)
        table.auto_set_font_size(False); table.set_fontsize(6); table.scale(0.8, 1.2)
        ax.set_title("Ranking (top 10)")
    # Plot 2: slope chart
    ax = axes[0,1]
    data = league_comparison.sort_values("reference_position")
    for _, row in data.iterrows():
        color = "#2ca25f" if row["abs_rank_difference"] <= 2 else "#de2d26"
        ax.plot([0,1], [row["reference_position"], row["simulated_position"]], color=color, alpha=0.7)
    ax.set_xlim(-0.2, 1.2)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Referência", "Simulação"])
    ax.set_ylabel("Posição"); ax.invert_yaxis()
    ax.set_title("Comparação de posições")
    # Plot 3: matriz de confusão
    ax = axes[1,0]
    labels = ["A_above_B", "A_below_B"]
    raw = np.zeros((2,2))
    for _, row in pairwise_df[pairwise_df["reason"].isin(["correct","wrong"])].iterrows():
        if row["reference_order"] in labels and row["simulated_order"] in labels:
            raw[labels.index(row["reference_order"]), labels.index(row["simulated_order"])] += 1
    matrix = raw
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_xticklabels(["A acima", "A abaixo"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["A acima", "A abaixo"])
    ax.set_xlabel("Simulado"); ax.set_ylabel("Referência")
    for i in range(2):
        for j in range(2):
            ax.text(j,i, f"{int(matrix[i,j])}", ha="center", va="center", color="black")
    ax.set_title("Matriz de confusão")
    # Plot 4: scatter
    ax = axes[1,1]
    valid = results_df[(results_df["error_status"]=="ok") & results_df["ranking_allowed"] & results_df["reference_value_normalized"].notna()]
    ax.scatter(valid["reference_value_normalized"], valid["simulated_break_temperature"], s=40)
    ax.set_xlabel("Referência"); ax.set_ylabel("T sim. (K)")
    ax.set_title("Referência vs simulação")
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def save_results(results_df, pairwise_df, summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "ranking_validation_results.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_comparison.csv", index=False)
    pd.DataFrame([{
        "metric": "ranking_difference",
        "value": summary.get("ranking_difference"),
        "ranking_similarity": summary.get("ranking_similarity"),
        "n_comparable_pairs": summary.get("n_comparable_pairs"),
        "n_wrong_pairs": summary.get("n_wrong_pairs"),
    }]).to_csv(output_dir / "ranking_difference_metric.csv", index=False)
    with open(output_dir / "ranking_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2)

def build_summary(results_df, pairwise_metrics, rank_correlations, error_metrics, league_data):
    side_by_side, mol_league, sim_league, league_comparison, league_metrics = league_data
    class_valid = results_df["class_correct"].dropna() if "class_correct" in results_df.columns else pd.Series(dtype=bool)
    rank_valid = results_df["rank_position_correct"].dropna() if "rank_position_correct" in results_df.columns else pd.Series(dtype=bool)
    n_cmp = pairwise_metrics.get("n_comparable_pairs", 0)
    n_wrg = pairwise_metrics.get("n_wrong_pairs", 0)
    ranking_diff = float(n_wrg / n_cmp) if n_cmp else None
    return {
        "n_molecules": len(results_df),
        "n_successful_simulations": int((results_df["error_status"] == "ok").sum()),
        "n_failed_simulations": int((results_df["error_status"] != "ok").sum()),
        "ranking_difference": ranking_diff,
        "ranking_similarity": pairwise_metrics.get("pairwise_accuracy"),
        **pairwise_metrics,
        **rank_correlations,
        "classification_accuracy": float(class_valid.mean()) if len(class_valid) else None,
        "rank_position_accuracy": float(rank_valid.mean()) if len(rank_valid) else None,
        "league_table_accuracy": float(side_by_side["position_match"].mean()) if len(side_by_side) else None,
        "league_metrics": league_metrics,
        "error_metrics": error_metrics,
        "reference_types_found": sorted(results_df["reference_type"].dropna().unique().astype(str)),
    }

def main():
    parser = argparse.ArgumentParser(description="Validação científica por ranking para MD")
    parser.add_argument("--input", default="../data/validation_molecules.csv", help="CSV com moléculas")
    parser.add_argument("--output", default="../results", help="Diretório de saída")
    parser.add_argument("--optuna", default="../data/optuna_resultado.json", help="JSON com parâmetros Optuna")
    parser.add_argument("--bde", default="../data/bde_results.csv", help="CSV/JSON com BDEs")
    parser.add_argument("--min-reference-gap", type=float, default=0.0)
    parser.add_argument("--min-simulated-gap", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bde_table = load_bde_table(args.bde)
    validation_df = load_validation_data(args.input)
    params = load_optuna_params(args.optuna)

    results_df, error_metrics = run_all_simulations(validation_df, params, bde_table)
    pairwise_df, pairwise_metrics = compute_pairwise_accuracy(results_df, args.min_reference_gap, args.min_simulated_gap)
    rank_correlations = compute_rank_correlations(results_df)
    league_data = build_league_tables(results_df)
    side_by_side, _, _, _, _ = league_data

    # Gerar gráficos
    plot_league_table(side_by_side, output_dir / "ranking_league_table.png")
    plot_league_position_comparison(league_data[3], output_dir / "league_position_comparison.png")
    plot_rank_difference(league_data[3], output_dir / "rank_difference_by_molecule.png")
    plot_scatter_ref_vs_sim(results_df, output_dir / "reference_vs_simulated.png")
    plot_pairwise_matrix(results_df, pairwise_df, output_dir / "pairwise_matrix.png")
    plot_confusion_matrix(pairwise_df, output_dir / "confusion_matrix.png", normalized=False)
    plot_confusion_matrix(pairwise_df, output_dir / "confusion_matrix_normalized.png", normalized=True)
    plot_dashboard(results_df, pairwise_df, league_data, output_dir / "validation_dashboard.png")

    summary = build_summary(results_df, pairwise_metrics, rank_correlations, error_metrics, league_data)
    save_results(results_df, pairwise_df, summary, output_dir)

    print("\n=== VALIDAÇÃO CONCLUÍDA ===")
    print(f"ranking_difference: {summary['ranking_difference']}")
    print(f"pairwise_accuracy: {summary['ranking_similarity']}")
    print(f"Resultados salvos em {output_dir}")

if __name__ == "__main__":
    main()