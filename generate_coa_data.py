"""
generate_coa_data.py
Ceramix Materials - 공급사 4곳의 비정형 합성 COA 40건, 과거 이력 DB 및 정답지 생성기
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from coa_tools import evaluate_lot_and_spc

SEED = 42
OUTPUT_DIR = "data"
COA_DIR = os.path.join(OUTPUT_DIR, "coa_samples")

SUPPLIERS: Dict[str, str] = {
    "SUP-A": "한빛세라믹(주)",
    "SUP-B": "Nippon Fine Powder Corp.",
    "SUP-C": "SinoMat Advanced Powders Ltd.",
    "SUP-D": "EuroOxide GmbH",
}


def render_coa_text(
    supplier_id: str,
    lot_id: str,
    date_str: str,
    vals: Dict[str, float],
    rng: np.random.Generator,
) -> Tuple[str, List[str]]:
    d50_um = vals["d50_um"]
    bet_m2g = vals["bet_m2g"]
    fe_ppm = vals["fe_ppm"]
    moist_wt = vals["moisture_wt"]
    traps: List[str] = []

    if supplier_id == "SUP-A":
        text = (
            f"================ 시험성적서 (Certificate of Analysis) ================\n"
            f"공급사: 한빛세라믹(주) | 품목명: CMX-YSZ-05 (고순도 산화물 분말)\n"
            f"Lot No.: {lot_id} | 출하일자: {date_str} | 수량: 1,000 kg\n"
            f"---------------------------------------------------------------------\n"
            f"검사항목                 | 규격(Spec)         | 측정결과      | 단위\n"
            f"---------------------------------------------------------------------\n"
            f"D50 (평균입도)           | 0.40 ~ 0.60        | {d50_um:.3f}         | um\n"
            f"BET (비표면적)           | 5.50 ~ 7.50        | {bet_m2g:.2f}          | m2/g\n"
            f"Fe (철 불순물)           | <= 100             | {fe_ppm:.1f}          | ppm\n"
            f"Moisture (수분함량)      | <= 0.50            | {moist_wt:.2f}          | wt%\n"
            f"---------------------------------------------------------------------\n"
            f"판정: 출하 기준 확인 완료 / 품질책임자: 김민수\n"
        )
        return text, traps

    if supplier_id == "SUP-B":
        d50_nm = round(d50_um * 1000.0, 1)
        fe_wt = round(fe_ppm / 10000.0, 5)
        d10_nm = round(d50_nm * 0.52, 1)
        d90_nm = round(d50_nm * 1.68, 1)
        traps = ["unit_nm_for_d50", "unit_wt%_for_fe", "alias_SSA_LOD"]
        text = (
            f"CERTIFICATE OF INSPECTION - NIPPON FINE POWDER CORP.\n"
            f"Material Code: NFP-YSZ5 (For Ceramix Materials)\n"
            f"Batch / Lot ID: {lot_id}   Date of Issue: {date_str}\n"
            f"Net Weight: 1200 kg   Packing: 25kg Fiber Drum x 48\n\n"
            f"[Physical Properties]\n"
            f"  * Particle Size Distribution (Laser Diffraction):\n"
            f"      D10 = {d10_nm} nm, Mean Particle Size (D50) = {d50_nm} nm, D90 = {d90_nm} nm\n"
            f"  * Specific Surface Area (SSA, N2 adsorption): {bet_m2g:.2f} m2/g\n\n"
            f"[Chemical Purity & Moisture]\n"
            f"  * Iron impurity (Fe, ICP-OES): {fe_wt:.5f} wt%\n"
            f"  * Sodium (Na): 0.0021 wt%   |   Silicon (Si): 0.0034 wt%\n"
            f"  * Loss on Drying (LOD, 110C 2hr): {moist_wt:.2f} %\n\n"
            f"QA Manager: T. Sato (Authorized Digital Signature)\n"
        )
        return text, traps

    if supplier_id == "SUP-C":
        moist_ppm = int(round(moist_wt * 10000.0))
        use_fe_wt = bool(rng.random() < 0.6)
        use_d50_nm = bool(rng.random() < 0.5)
        d50_str = f"{round(d50_um * 1000.0, 1)} nm" if use_d50_nm else f"{d50_um:.3f} micron"
        fe_str = f"{fe_ppm / 10000.0:.5f} wt%" if use_fe_wt else f"{fe_ppm:.1f} mg/kg"
        traps = ["unstructured_email", "unit_ppm_for_moisture"]
        if use_fe_wt:
            traps.append("unit_wt%_for_fe")
        if use_d50_nm:
            traps.append("unit_nm_for_d50")

        text = (
            f"From: qc_export@sinomat-powder.cn\n"
            f"Subject: Lab Test Report for Lot {lot_id} (Shipment {date_str})\n\n"
            f"Dear Ceramix IQC Team,\n"
            f"Please find below the final lab inspection data for 800 kg of CMX-YSZ-05 powder.\n"
            f"Laser particle analyzer showed D10 at 0.24 um, Median Dia. (d50) = {d50_str}, "
            f"and D90 = 0.92 um. BET Surface measured via 5-point method came out to {bet_m2g:.2f} m^2/g.\n"
            f"For trace elements by ICP: Al 18 mg/kg, Na 24 mg/kg, and Fe impurity is {fe_str}.\n"
            f"Karl-Fischer H2O content before vacuum sealing was {moist_ppm} ppm.\n"
            f"(Reference customer limits: D50 0.40-0.60 um, BET 5.50-7.50 m2/g, Fe max 100 ppm)\n"
            f"Best regards,\n"
            f"Li Wei, SinoMat QC Dept.\n"
        )
        return text, traps

    use_d50_nm = bool(rng.random() < 0.5)
    d50_display = f"{round(d50_um * 1000.0, 1)} | nm" if use_d50_nm else f"{d50_um:.3f} | μm"
    fe_display = f"{fe_ppm / 10000.0:.5f} | wt%" if fe_ppm > 80 else f"{fe_ppm:.1f} | ppm"
    traps = ["pipe_table_adjacent_spec", "alias_d(0.5)_residual_moisture"]
    if use_d50_nm:
        traps.append("unit_nm_for_d50")
    if fe_ppm > 80:
        traps.append("unit_wt%_for_fe")

    text = (
        f"EUROOXIDE GMBH - ANALYTICAL QUALITY REPORT (DIN EN 10204 3.1)\n"
        f"Batch No: {lot_id} ; Release Date: {date_str} ; Product: Ceramic Powder YSZ-05\n"
        f"Parameter | Customer Spec | Measured Result | Unit\n"
        f"Particle Size d(0.5) | 0.40 - 0.60 um | {d50_display}\n"
        f"Surface Area (N2-BET) | 5.50 - 7.50 m2/g | {bet_m2g:.2f} | m²/g\n"
        f"Fe content (ICP) | max 100 ppm | {fe_display}\n"
        f"Residual Moisture | max 0.50 wt% | {moist_wt:.2f} | wt.%\n"
        f"Tap Density | min 1.10 g/cm3 | 1.24 | g/cm3\n"
        f"Approved by: Dr. H. Weber, Quality Assurance\n"
    )
    return text, traps


def _gen_normal_values(rng: np.random.Generator, sign_hint: int = 0) -> Dict[str, float]:
    d50_delta = abs(float(rng.normal(0.0, 0.014)))
    bet_delta = abs(float(rng.normal(0.0, 0.11)))
    if sign_hint == 0:
        d50_sign = 1.0 if rng.random() < 0.5 else -1.0
        bet_sign = 1.0 if rng.random() < 0.5 else -1.0
    else:
        d50_sign = float(sign_hint)
        bet_sign = float(-sign_hint)

    return {
        "d50_um": round(float(np.clip(0.50 + d50_sign * d50_delta, 0.45, 0.55)), 3),
        "bet_m2g": round(float(np.clip(6.50 + bet_sign * bet_delta, 6.15, 6.85)), 2),
        "fe_ppm": round(float(np.clip(rng.normal(45.0, 7.5), 25.0, 68.0)), 1),
        "moisture_wt": round(float(np.clip(rng.normal(0.22, 0.035), 0.12, 0.34)), 2),
    }


def generate_dataset() -> None:
    rng = np.random.default_rng(SEED)
    os.makedirs(COA_DIR, exist_ok=True)

    history_rows: List[Dict[str, Any]] = []
    for sup_id, sup_name in SUPPLIERS.items():
        for h_idx in range(1, 11):
            sign = 1 if h_idx % 2 == 0 else -1
            vals = _gen_normal_values(rng, sign_hint=sign)
            history_rows.append(
                {
                    "lot_id": f"HIST-{sup_id[-1]}-{h_idx:02d}",
                    "supplier_id": sup_id,
                    "supplier_name": sup_name,
                    "date": f"2026-09-{h_idx:02d}",
                    **vals,
                    "status": "PASS",
                }
            )

    history_seed_df = pd.DataFrame(history_rows)
    history_seed_path = os.path.join(OUTPUT_DIR, "history_seed.csv")
    history_seed_df.to_csv(history_seed_path, index=False, encoding="utf-8-sig")

    planned_lots: List[Dict[str, Any]] = []

    sup_a_fail_map = {
        3: ("fe_ppm", 132.5, "spec_fail_fe"),
        6: ("moisture_wt", 0.68, "spec_fail_moisture"),
        8: ("d50_um", 0.665, "spec_fail_d50_high"),
        10: ("bet_m2g", 5.12, "spec_fail_bet_low"),
    }
    for i in range(1, 11):
        vals = _gen_normal_values(rng, sign_hint=1 if i % 2 == 0 else -1)
        scenario = "normal_standard"
        if i in sup_a_fail_map:
            col, bad_val, scenario = sup_a_fail_map[i]
            vals[col] = bad_val
        planned_lots.append({"supplier_id": "SUP-A", "seq": i, "vals": vals, "scenario": scenario})

    bet_drift_seq = [6.46, 6.56, 6.64, 6.72, 6.81, 6.90, 6.99, 7.09, 7.19, 7.31]
    for i in range(1, 11):
        vals = _gen_normal_values(rng, sign_hint=1 if i % 2 == 0 else -1)
        vals["bet_m2g"] = bet_drift_seq[i - 1]
        scenario = "normal_jp_units" if i <= 6 else "spc_drift_bet_up"
        planned_lots.append({"supplier_id": "SUP-B", "seq": i, "vals": vals, "scenario": scenario})

    sup_c_fail_map = {
        2: ("d50_um", 0.655, "spec_fail_d50_nm_trap"),
        4: ("fe_ppm", 145.0, "spec_fail_fe_wt_trap"),
        6: ("moisture_wt", 0.64, "spec_fail_moisture_ppm_trap"),
        8: ("bet_m2g", 7.88, "spec_fail_bet_high"),
        10: ("d50_um", 0.345, "spec_fail_d50_low"),
    }
    for i in range(1, 11):
        vals = _gen_normal_values(rng, sign_hint=1 if i % 2 == 0 else -1)
        scenario = "normal_cn_email"
        if i in sup_c_fail_map:
            col, bad_val, scenario = sup_c_fail_map[i]
            vals[col] = bad_val
        planned_lots.append({"supplier_id": "SUP-C", "seq": i, "vals": vals, "scenario": scenario})

    d50_drift_seq = [0.502, 0.494, 0.487, 0.479, 0.471, 0.463, 0.454, 0.444, 0.434, 0.422]
    for i in range(1, 11):
        vals = _gen_normal_values(rng, sign_hint=1 if i % 2 == 0 else -1)
        if i == 1:
            vals["fe_ppm"] = 128.0
            scenario = "spec_fail_de_fe_wt_trap"
        else:
            vals["d50_um"] = d50_drift_seq[i - 1]
            scenario = "normal_de_pipe" if i <= 6 else "spc_drift_d50_down"
        planned_lots.append({"supplier_id": "SUP-D", "seq": i, "vals": vals, "scenario": scenario})

    rolling_history: Dict[str, pd.DataFrame] = {
        sup_id: history_seed_df[history_seed_df["supplier_id"] == sup_id].copy()
        for sup_id in SUPPLIERS
    }

    gt_records: List[Dict[str, Any]] = []
    for idx, item in enumerate(planned_lots, start=1):
        sup_id = item["supplier_id"]
        sup_name = SUPPLIERS[sup_id]
        lot_id = f"LOT-2026-{idx:03d}"
        coa_filename = f"COA_{idx:03d}.txt"
        coa_path = os.path.join(COA_DIR, coa_filename).replace("\\", "/")
        day = 1 + ((idx - 1) % 28)
        month = 10 if idx <= 28 else 11
        date_str = f"2026-{month:02d}-{day:02d}"

        coa_text, traps = render_coa_text(sup_id, lot_id, date_str, item["vals"], rng)
        with open(coa_path, "w", encoding="utf-8") as f:
            f.write(coa_text)

        eval_gt = evaluate_lot_and_spc(item["vals"], rolling_history[sup_id])

        gt_records.append(
            {
                "coa_id": f"COA_{idx:03d}",
                "lot_id": lot_id,
                "date": date_str,
                "supplier_id": sup_id,
                "supplier_name": sup_name,
                "coa_path": coa_path,
                "scenario_tag": item["scenario"],
                "traps": traps,
                "true_values": item["vals"],
                "true_status": eval_gt["final_status"],
                "true_spec_violations": [v["metric"] for v in eval_gt["spec_violations"]],
                "true_spc_alert_metrics": list(
                    dict.fromkeys(a["metric"] for a in eval_gt["spc_alerts"])
                ),
                "true_bet_cpk": eval_gt["spc_metrics"]["bet_m2g"]["cpk"],
                "true_d50_cpk": eval_gt["spc_metrics"]["d50_um"]["cpk"],
            }
        )

        if eval_gt["final_status"] != "FAIL":
            new_row = pd.DataFrame(
                [
                    {
                        "lot_id": lot_id,
                        "supplier_id": sup_id,
                        "supplier_name": sup_name,
                        "date": date_str,
                        **item["vals"],
                        "status": eval_gt["final_status"],
                    }
                ]
            )
            rolling_history[sup_id] = pd.concat(
                [rolling_history[sup_id], new_row], ignore_index=True
            )

    gt_json_path = os.path.join(OUTPUT_DIR, "ground_truth.json")
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(gt_records, f, ensure_ascii=False, indent=2)

    flat_rows = []
    for r in gt_records:
        flat_rows.append(
            {
                "coa_id": r["coa_id"],
                "lot_id": r["lot_id"],
                "supplier_id": r["supplier_id"],
                "scenario_tag": r["scenario_tag"],
                "d50_um": r["true_values"]["d50_um"],
                "bet_m2g": r["true_values"]["bet_m2g"],
                "fe_ppm": r["true_values"]["fe_ppm"],
                "moisture_wt": r["true_values"]["moisture_wt"],
                "true_status": r["true_status"],
                "true_bet_cpk": r["true_bet_cpk"],
                "true_d50_cpk": r["true_d50_cpk"],
            }
        )
    gt_csv_path = os.path.join(OUTPUT_DIR, "ground_truth.csv")
    pd.DataFrame(flat_rows).to_csv(gt_csv_path, index=False, encoding="utf-8-sig")

    print(f"총 {len(gt_records)}건의 비정형 COA 생성 완료 -> {COA_DIR}")
    print("\n[공급사별 / 최종 판정별 분포]")
    df_sum = pd.DataFrame(flat_rows)
    print(pd.crosstab(df_sum["supplier_id"], df_sum["true_status"], margins=True).to_string())


if __name__ == "__main__":
    generate_dataset()
