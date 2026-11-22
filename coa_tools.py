"""
coa_tools.py
Ceramix Materials - 세라믹 원료 분말(CMX-YSZ-05) COA 단위 환산, 물리 검증, SPC 통계 및 SCAR 생성 툴
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SPEC_RULES: Dict[str, Dict[str, Any]] = {
    "d50_um": {
        "label": "평균 입도 (D50)",
        "unit": "um",
        "lsl": 0.40,
        "target": 0.50,
        "usl": 0.60,
        "two_sided": True,
    },
    "bet_m2g": {
        "label": "비표면적 (BET/SSA)",
        "unit": "m2/g",
        "lsl": 5.50,
        "target": 6.50,
        "usl": 7.50,
        "two_sided": True,
    },
    "fe_ppm": {
        "label": "철 불순물 (Fe)",
        "unit": "ppm",
        "lsl": None,
        "target": 45.0,
        "usl": 100.0,
        "two_sided": False,
    },
    "moisture_wt": {
        "label": "수분 함량 (Moisture)",
        "unit": "wt%",
        "lsl": None,
        "target": 0.22,
        "usl": 0.50,
        "two_sided": False,
    },
}

PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "d50_um": (0.05, 50.0),
    "bet_m2g": (0.50, 100.0),
    "fe_ppm": (1.0, 5000.0),
    "moisture_wt": (0.01, 5.0),
}

PROCESS_IMPACT_KB: Dict[str, str] = {
    "d50_um_high": "입도 조대화로 비표면적이 저하되어 소결 구동력 감소 및 최종 소결체 밀도 미달 위험.",
    "d50_um_low": "미분 증가로 입자 간 응집(agglomeration) 심화 및 성형 밀도 변화에 따른 소결 수축률 편차 발생 위험.",
    "bet_m2g_high": "비표면적 상승으로 표면 자유에너지 증가 -> 슬러리 점도 급상승, 분산 불량 및 내부 기공 결함 유발 위험.",
    "bet_m2g_low": "비표면적 저하로 소결 활성도 감소 -> 동일 소결 프로파일에서 치밀화 불량 우려.",
    "fe_ppm_high": "미량 금속 불순물(Fe)의 결정립계 편석으로 이상 입자 성장(abnormal grain growth) 및 절연·전기적 특성 저하 위험.",
    "moisture_wt_high": "잔류 수분 과다로 분말 케이킹(caking) 및 성형체 건조·탈지 과정 중 미세 균열(crack) 유발 위험.",
}


def _clean_unit(unit: str) -> str:
    return re.sub(r"\s+", "", str(unit or "")).lower()


def normalize_unit(metric: str, value: float, raw_unit: str) -> Tuple[Optional[float], str]:
    """공급사별 상이한 단위를 사내 표준 단위(um, m2/g, ppm, wt%)로 강제 환산."""
    u = _clean_unit(raw_unit)

    if metric == "d50_um":
        if u in ("um", "μm", "µm", "micron", "microns", "micrometer", "micrometers"):
            return round(value, 5), "um"
        if u in ("nm", "nanometer", "nanometers"):
            return round(value / 1000.0, 5), "um (nm->um 환산)"
        if u in ("mm",):
            return round(value * 1000.0, 5), "um (mm->um 환산)"
        return None, f"알 수 없는 입도 단위: '{raw_unit}'"

    if metric == "bet_m2g":
        if u in ("m2/g", "m^2/g", "m²/g", "m2g-1", "m2·g-1", "m2/gm"):
            return round(value, 5), "m2/g"
        if u in ("cm2/g", "cm^2/g", "cm²/g"):
            return round(value / 10000.0, 5), "m2/g (cm2/g->m2/g 환산)"
        return None, f"알 수 없는 비표면적 단위: '{raw_unit}'"

    if metric == "fe_ppm":
        if u in ("ppm", "mg/kg", "ug/g", "μg/g", "µg/g"):
            return round(value, 4), f"ppm ({raw_unit}->ppm)"
        if u in ("wt%", "%", "wt.%", "mass%", "percent"):
            return round(value * 10000.0, 4), "ppm (wt%->ppm 환산)"
        if u in ("ppb",):
            return round(value / 1000.0, 4), "ppm (ppb->ppm 환산)"
        return None, f"알 수 없는 불순물 단위: '{raw_unit}'"

    if metric == "moisture_wt":
        if u in ("wt%", "%", "wt.%", "mass%", "percent"):
            return round(value, 5), "wt%"
        if u in ("ppm", "mg/kg", "ug/g", "μg/g"):
            return round(value / 10000.0, 5), "wt% (ppm->wt% 환산)"
        return None, f"알 수 없는 수분 단위: '{raw_unit}'"

    return None, f"지원하지 않는 항목: {metric}"


def validate_and_standardize_coa(raw_extraction: Dict[str, Any]) -> Dict[str, Any]:
    """추출된 JSON을 표준 단위로 변환하고 물리적 타당성 검증 후 수정 힌트 반환."""
    standardized: Dict[str, Optional[float]] = {}
    conversion_log: Dict[str, str] = {}
    issues: List[str] = []
    hints: List[str] = []

    for metric in SPEC_RULES:
        entry = raw_extraction.get(metric)
        if not isinstance(entry, dict) or entry.get("value") is None:
            issues.append(f"{metric} 값 누락")
            hints.append(f"원문에서 {SPEC_RULES[metric]['label']} 측정값과 단위를 다시 찾으세요 (규격 범위가 아닌 실제 측정 결과값 추출).")
            standardized[metric] = None
            continue

        try:
            raw_val = float(entry["value"])
        except (TypeError, ValueError):
            issues.append(f"{metric} 숫자 변환 실패: {entry.get('value')}")
            hints.append(f"{metric}의 순수 숫자만 추출하세요.")
            standardized[metric] = None
            continue

        raw_unit = str(entry.get("unit", "")).strip()
        if not raw_unit:
            issues.append(f"{metric} 단위 누락 (추출값={raw_val})")
            hints.append(f"원문에서 {metric} 값({raw_val}) 옆의 단위(예: nm/um, wt%/ppm)를 정확히 읽으세요.")
            standardized[metric] = None
            continue

        std_val, note = normalize_unit(metric, raw_val, raw_unit)
        if std_val is None:
            issues.append(f"{metric} 단위 변환 실패: {note}")
            hints.append(f"{metric}의 단위 표기를 원문에서 재확인하세요.")
            standardized[metric] = None
            continue

        standardized[metric] = std_val
        conversion_log[metric] = f"{raw_val} {raw_unit} -> {std_val} [{note}]"

        low, high = PHYSICAL_BOUNDS[metric]
        if not (low <= std_val <= high):
            issues.append(
                f"{metric} 물리적 타당 범위({low}~{high}) 이탈: 환산값={std_val} (원문={raw_val} {raw_unit})"
            )
            if metric == "d50_um" and std_val > high:
                hints.append("입도 D50이 비정상적으로 큽니다. 원문 단위가 'nm'인데 'um'로 잘못 읽지 않았는지 확인하세요.")
            elif metric == "fe_ppm" and std_val < low:
                hints.append("Fe 함량이 1 ppm 미만입니다. 원문 단위가 'wt%'인데 'ppm'으로 잘못 읽지 않았는지 확인하세요.")
            elif metric == "moisture_wt" and std_val > high:
                hints.append("수분 함량이 5 wt%를 초과합니다. 원문 단위가 'ppm'인데 'wt%'로 잘못 읽지 않았는지 확인하세요.")
            else:
                hints.append(f"{metric} 수치가 물리 범위를 벗어났습니다. 표의 인접 열이나 규격 숫자를 잘못 읽었는지 재확인하세요.")

    return {
        "valid": len(issues) == 0,
        "standardized_values": standardized,
        "conversion_log": conversion_log,
        "issues": issues,
        "correction_hints": list(dict.fromkeys(hints)),
    }


def _calc_cpk(values: np.ndarray, lsl: Optional[float], usl: float) -> float:
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if std < 1e-9:
        return 9.99
    cpu = (usl - mean) / (3.0 * std)
    if lsl is None:
        return round(float(cpu), 3)
    cpl = (mean - lsl) / (3.0 * std)
    return round(float(min(cpu, cpl)), 3)


def evaluate_lot_and_spc(
    standardized_values: Dict[str, float],
    supplier_history_df: pd.DataFrame,
) -> Dict[str, Any]:
    """단건 규격 판정(FAIL)과 공급사 누적 이력 기반 Cpk·추세 드리프트(SPC_WARNING) 계산."""
    spec_violations: List[Dict[str, Any]] = []
    spc_alerts: List[Dict[str, Any]] = []
    spc_metrics: Dict[str, Dict[str, Any]] = {}
    process_impacts: List[str] = []

    hist_tail = supplier_history_df.tail(19) if len(supplier_history_df) > 0 else pd.DataFrame()

    for metric, rule in SPEC_RULES.items():
        val = float(standardized_values[metric])
        lsl = rule["lsl"]
        usl = rule["usl"]
        target = rule["target"]

        out_high = val > usl
        out_low = (lsl is not None) and (val < lsl)
        if out_high:
            spec_violations.append(
                {"metric": metric, "value": val, "limit": f"USL {usl}", "direction": "HIGH"}
            )
            key = f"{metric}_high"
            if key in PROCESS_IMPACT_KB:
                process_impacts.append(PROCESS_IMPACT_KB[key])
        elif out_low:
            spec_violations.append(
                {"metric": metric, "value": val, "limit": f"LSL {lsl}", "direction": "LOW"}
            )
            key = f"{metric}_low"
            if key in PROCESS_IMPACT_KB:
                process_impacts.append(PROCESS_IMPACT_KB[key])

        hist_vals = hist_tail[metric].to_numpy(dtype=float) if metric in hist_tail.columns else np.array([])
        series = np.append(hist_vals, val)

        mean_val = round(float(np.mean(series)), 4)
        std_val = round(float(np.std(series, ddof=1)), 4) if len(series) > 1 else 0.0
        cpk = _calc_cpk(series, lsl, usl)

        run7_high = False
        run7_low = False
        trend6_up = False
        trend6_down = False

        if rule["two_sided"]:
            if len(series) >= 7:
                last7 = series[-7:]
                run7_high = bool(np.all(last7 > target))
                run7_low = bool(np.all(last7 < target))
            if len(series) >= 6:
                last6 = series[-6:]
                trend6_up = bool(np.all(np.diff(last6) > 0))
                trend6_down = bool(np.all(np.diff(last6) < 0))

            if run7_high or trend6_up:
                reason = []
                if run7_high:
                    reason.append("7연속 중심선 상향 치우침(Run-7 High)")
                if trend6_up:
                    reason.append("6연속 단조 상승 추세(Trend-6 Up)")
                spc_alerts.append(
                    {"metric": metric, "cpk": cpk, "current_value": val, "reasons": reason}
                )
                key = f"{metric}_high"
                if key in PROCESS_IMPACT_KB:
                    process_impacts.append(f"[선제 경고] {PROCESS_IMPACT_KB[key]}")

            if run7_low or trend6_down:
                reason = []
                if run7_low:
                    reason.append("7연속 중심선 하향 치우침(Run-7 Low)")
                if trend6_down:
                    reason.append("6연속 단조 하락 추세(Trend-6 Down)")
                spc_alerts.append(
                    {"metric": metric, "cpk": cpk, "current_value": val, "reasons": reason}
                )
                key = f"{metric}_low"
                if key in PROCESS_IMPACT_KB:
                    process_impacts.append(f"[선제 경고] {PROCESS_IMPACT_KB[key]}")

        if cpk < 1.00 and not (out_high or out_low):
            spc_alerts.append(
                {
                    "metric": metric,
                    "cpk": cpk,
                    "current_value": val,
                    "reasons": [f"공정능력지수 저하 (Cpk={cpk:.2f} < 1.00)"],
                }
            )

        spc_metrics[metric] = {
            "current": val,
            "mean_20": mean_val,
            "std_20": std_val,
            "cpk": cpk,
            "run7_high": run7_high,
            "run7_low": run7_low,
            "trend6_up": trend6_up,
            "trend6_down": trend6_down,
        }

    if spec_violations:
        final_status = "FAIL"
        decision_summary = "규격 이탈(Spec Out) 감지 -> 즉시 입고 보류(Quarantine) 및 공급사 SCAR 발행 필요."
    elif spc_alerts:
        final_status = "SPC_WARNING"
        decision_summary = "단건 규격은 충족하나 누적 공정 드리프트(SPC 이상) 감지 -> 조건부 입고 및 선제 시정조치(SCAR) 검토."
    else:
        final_status = "PASS"
        decision_summary = "전 항목 규격 적합 및 공급사 공정 안정(SPC 정상) -> 정상 입고 승인."

    return {
        "final_status": final_status,
        "decision_summary": decision_summary,
        "spec_violations": spec_violations,
        "spc_alerts": spc_alerts,
        "spc_metrics": spc_metrics,
        "process_impacts": list(dict.fromkeys(process_impacts)),
    }


def build_scar_draft(
    lot_id: str,
    supplier_name: str,
    eval_result: Dict[str, Any],
) -> str:
    """FAIL 또는 SPC_WARNING 판정 시 공급사 회신용 SCAR 초안 텍스트 반환."""
    status = eval_result["final_status"]
    if status == "PASS":
        return f"[{lot_id}] 정상 입고 승인(PASS) 건으로 SCAR 발행 대상이 아닙니다."

    lines = [
        "=== 공급사 시정조치요구서 (SCAR) 초안 ===",
        "- 발행처: Ceramix Materials 품질보증팀 (IQC)",
        f"- 수신처: {supplier_name}",
        f"- 대상 로트 / 품목: {lot_id} / CMX-YSZ-05 고순도 세라믹 원료 분말",
        f"- 판정 등급: {status}",
        "",
        "[1. 이상 감지 내역]",
    ]
    for v in eval_result["spec_violations"]:
        lines.append(
            f"  * 규격 이탈: {SPEC_RULES[v['metric']]['label']} = {v['value']} ({v['limit']} {v['direction']})"
        )
    for a in eval_result["spc_alerts"]:
        lines.append(
            f"  * 공정 추세 이상: {SPEC_RULES[a['metric']]['label']} = {a['current_value']} (Cpk={a['cpk']}, 사유: {', '.join(a['reasons'])})"
        )

    lines.append("")
    lines.append("[2. 당사 후공정(슬러리·소결) 예상 영향]")
    for imp in eval_result["process_impacts"]:
        lines.append(f"  * {imp}")

    lines.append("")
    lines.append("[3. 공급사 요청 조치 (회신 기한: 영업일 3일 이내)]")
    if status == "FAIL":
        lines.append("  1) 해당 로트 전량 반품/교체 출하 및 긴급 대체 로트 COA 선송부")
        lines.append("  2) 원료 분쇄·열처리·포장 공정의 근본 원인 분석(Root Cause) 및 8D 리포트 제출")
    else:
        lines.append("  1) 최근 연속 로트 물성 편향 원인(밀링 미디어 마모, 원료 로트 변경 등) 점검 결과 회신")
        lines.append("  2) 차기 출하 로트부터 목표 중심값 복귀를 위한 공정 조건 조정 계획 제출")

    return "\n".join(lines)


if __name__ == "__main__":
    print("coa_tools.py 로드 성공. 관리 항목:", list(SPEC_RULES.keys()))
