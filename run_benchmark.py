"""
run_benchmark.py
40건 비정형 COA 데이터셋 대상 Baseline vs Agent v1 vs Agent v2 정량 비교 실험 및 그래프 생성
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_coa import (
    get_llm_client,
    run_agent_v1_simple_llm,
    run_agent_v2_self_correcting,
    run_baseline_regex,
)
from coa_tools import SPEC_RULES

RESULTS_DIR = "results"


def _values_match(
    pred: Dict[str, Any], true_vals: Dict[str, float], rel_tol: float = 0.02
) -> bool:
    for metric, t_val in true_vals.items():
        p_val = pred.get(metric)
        if p_val is None:
            return False
        if abs(float(p_val) - float(t_val)) > max(abs(float(t_val)) * rel_tol, 1e-3):
            return False
    return True


def _summarize_method(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    total = len(records)
    parse_ok = sum(1 for r in records if r[key]["parse_success"])
    val_ok = sum(1 for r in records if r[key]["value_exact"])
    status_ok = sum(1 for r in records if r[key]["status_exact"])

    fail_cases = [r for r in records if r["true_status"] == "FAIL"]
    fail_detected = sum(1 for r in fail_cases if r[key]["predicted_status"] == "FAIL")

    spc_cases = [r for r in records if r["true_status"] == "SPC_WARNING"]
    spc_detected = sum(1 for r in spc_cases if r[key]["predicted_status"] == "SPC_WARNING")

    avg_latency = float(np.mean([r[key]["latency_sec"] for r in records]))
    total_retries = sum(int(r[key].get("retries_used", 0)) for r in records)

    return {
        "Method": key,
        "Total_Samples": total,
        "Parse_Success_Rate_pct": round(parse_ok / total * 100.0, 1),
        "Value_Unit_Accuracy_pct": round(val_ok / total * 100.0, 1),
        "Overall_Decision_Accuracy_pct": round(status_ok / total * 100.0, 1),
        "Spec_FAIL_Recall_pct": round(fail_detected / max(len(fail_cases), 1) * 100.0, 1),
        "SPC_Drift_Recall_pct": round(spc_detected / max(len(spc_cases), 1) * 100.0, 1),
        "Avg_Latency_sec": round(avg_latency, 3),
        "Total_Self_Retries": total_retries,
    }


def _plot_benchmark_chart(summary_df: pd.DataFrame, out_path: str) -> None:
    metrics = [
        ("Parse_Success_Rate_pct", "Parse\nSuccess"),
        ("Value_Unit_Accuracy_pct", "Unit & Value\nAccuracy"),
        ("Overall_Decision_Accuracy_pct", "Overall\nDecision Acc"),
        ("Spec_FAIL_Recall_pct", "Spec FAIL\nRecall"),
        ("SPC_Drift_Recall_pct", "SPC Drift\nDetection"),
    ]
    labels = [m[1] for m in metrics]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=180)
    colors = ["#94A3B8", "#3B82F6", "#10B981"]
    display_names = {
        "baseline": "Baseline (Fixed Regex)",
        "agent_v1": "Agent v1 (Single LLM)",
        "agent_v2": "Agent v2 (Tool + Self-Check + SPC)",
    }

    for idx, method_key in enumerate(["baseline", "agent_v1", "agent_v2"]):
        row = summary_df[summary_df["Method"] == method_key].iloc[0]
        vals = [row[m[0]] for m in metrics]
        bars = ax.bar(x + (idx - 1) * width, vals, width, label=display_names[method_key], color=colors[idx])
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                v + 1.5,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("Ceramix Materials - 40 COA Benchmark: Baseline vs Agent v1 vs Agent v2", fontsize=12.5, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 114)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def _plot_spc_drift_chart(
    history_seed_df: pd.DataFrame,
    gt_records: List[Dict[str, Any]],
    supplier_id: str,
    metric: str,
    title: str,
    ylabel: str,
    out_path: str,
) -> None:
    rule = SPEC_RULES[metric]
    hist = history_seed_df[history_seed_df["supplier_id"] == supplier_id]
    new_lots = [
        r for r in gt_records
        if r["supplier_id"] == supplier_id and r["true_status"] != "FAIL"
    ]

    lot_labels = list(hist["lot_id"]) + [r["lot_id"] for r in new_lots]
    values = list(hist[metric]) + [r["true_values"][metric] for r in new_lots]
    statuses = ["HIST"] * len(hist) + [r["true_status"] for r in new_lots]

    x = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=180)
    ax.plot(x, values, color="#2563EB", linewidth=2.0, marker="o", markersize=5.5, label="Measured Lot Value")

    warn_x = [xi for xi, st in zip(x, statuses) if st == "SPC_WARNING"]
    warn_y = [yi for yi, st in zip(values, statuses) if st == "SPC_WARNING"]
    if warn_x:
        ax.scatter(warn_x, warn_y, color="#F97316", s=85, zorder=5, label="Agent v2 SPC Drift Alert (Within Spec!)")

    ax.axhline(rule["usl"], color="#DC2626", linestyle="--", linewidth=1.6, label=f"USL ({rule['usl']})")
    if rule["lsl"] is not None:
        ax.axhline(rule["lsl"], color="#DC2626", linestyle="--", linewidth=1.6, label=f"LSL ({rule['lsl']})")
    ax.axhline(rule["target"], color="#64748B", linestyle=":", linewidth=1.4, label=f"Target ({rule['target']})")

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Incoming Lot Sequence (History 1~10 + New Lots 11~)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl.replace("LOT-2026-", "L") for lbl in lot_labels], rotation=45, ha="right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gt_records = json.load(open("data/ground_truth.json", encoding="utf-8"))
    history_seed_df = pd.read_csv("data/history_seed.csv")
    client = get_llm_client()

    rolling_history: Dict[str, pd.DataFrame] = {
        sup_id: history_seed_df[history_seed_df["supplier_id"] == sup_id].copy()
        for sup_id in history_seed_df["supplier_id"].unique()
    }

    detailed_results: List[Dict[str, Any]] = []
    print(f"총 {len(gt_records)}건 COA 벤치마크 실행 시작 (Baseline / Agent v1 / Agent v2)...")

    for idx, rec in enumerate(gt_records, start=1):
        sup_id = rec["supplier_id"]
        coa_text = open(rec["coa_path"], encoding="utf-8").read()
        sup_hist = rolling_history[sup_id]
        true_vals = rec["true_values"]
        true_status = rec["true_status"]

        res_base = run_baseline_regex(coa_text)
        res_v1 = run_agent_v1_simple_llm(coa_text, client=client)
        res_v2 = run_agent_v2_self_correcting(
            coa_text=coa_text,
            supplier_history_df=sup_hist,
            lot_id=rec["lot_id"],
            supplier_name=rec["supplier_name"],
            client=client,
            generate_scar=True,
        )

        for res in (res_base, res_v1, res_v2):
            res["value_exact"] = _values_match(res["extracted_values"], true_vals)
            res["status_exact"] = res["predicted_status"] == true_status

        detailed_results.append(
            {
                "coa_id": rec["coa_id"],
                "lot_id": rec["lot_id"],
                "date": rec["date"],
                "supplier_id": sup_id,
                "supplier_name": rec["supplier_name"],
                "coa_path": rec["coa_path"],
                "scenario_tag": rec["scenario_tag"],
                "traps": rec["traps"],
                "true_values": true_vals,
                "true_status": true_status,
                "baseline": res_base,
                "agent_v1": res_v1,
                "agent_v2": res_v2,
            }
        )

        print(
            f"[{idx:02d}/40] {rec['coa_id']} ({sup_id}) | 정답={true_status:<11} | "
            f"Base={res_base['predicted_status']:<11} | v1={res_v1['predicted_status']:<11} | "
            f"v2={res_v2['predicted_status']:<11} (retry={res_v2['retries_used']})"
        )

        if true_status != "FAIL":
            new_row = pd.DataFrame(
                [
                    {
                        "lot_id": rec["lot_id"],
                        "supplier_id": sup_id,
                        "supplier_name": rec["supplier_name"],
                        "date": rec["date"],
                        **true_vals,
                        "status": true_status,
                    }
                ]
            )
            rolling_history[sup_id] = pd.concat([rolling_history[sup_id], new_row], ignore_index=True)

    summary_rows = [
        _summarize_method(detailed_results, "baseline"),
        _summarize_method(detailed_results, "agent_v1"),
        _summarize_method(detailed_results, "agent_v2"),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "benchmark_summary.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    with open(os.path.join(RESULTS_DIR, "detailed_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    _plot_benchmark_chart(summary_df, os.path.join(RESULTS_DIR, "benchmark_comparison.png"))
    _plot_spc_drift_chart(
        history_seed_df,
        gt_records,
        supplier_id="SUP-B",
        metric="bet_m2g",
        title="SPC Control Chart - SUP-B (Nippon Fine Powder): BET Upward Drift Inside Spec",
        ylabel="BET Specific Surface Area (m^2/g)",
        out_path=os.path.join(RESULTS_DIR, "spc_drift_sup_b_bet.png"),
    )
    _plot_spc_drift_chart(
        history_seed_df,
        gt_records,
        supplier_id="SUP-D",
        metric="d50_um",
        title="SPC Control Chart - SUP-D (EuroOxide GmbH): D50 Downward Drift Inside Spec",
        ylabel="Mean Particle Size D50 (um)",
        out_path=os.path.join(RESULTS_DIR, "spc_drift_sup_d_d50.png"),
    )

    print("\n================ 40건 벤치마크 최종 요약표 ================")
    print(summary_df.to_string(index=False))
    print(f"\n저장 완료: {RESULTS_DIR}/benchmark_summary.csv, detailed_predictions.json, PNG 그래프 3종")


if __name__ == "__main__":
    main()
