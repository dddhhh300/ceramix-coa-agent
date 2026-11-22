"""
agent_coa.py
Ceramix Materials - Baseline(정규식), Agent v1(단순 LLM), Agent v2(자가수정·SPC 툴 에이전트) 구현
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from coa_tools import (
    SPEC_RULES,
    build_scar_draft,
    evaluate_lot_and_spc,
    validate_and_standardize_coa,
)

load_dotenv()


def get_llm_client() -> OpenAI:
    """OpenAI 공식 API 또는 학교 OpenAI 호환 프록시(base_url) 클라이언트 생성."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(".env 파일에 OPENAI_API_KEY가 설정되어 있지 않습니다.")

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


def _extract_json_object(text: str) -> Dict[str, Any]:
    """LLM 응답 문자열에서 JSON 객체만 안전하게 파싱."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _chat_json(
    client: OpenAI,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """JSON 응답을 요청하되, 학교 프록시가 response_format을 미지원하면 일반 호출로 자동 폴백."""
    model_name = model or get_model_name()
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
    content = resp.choices[0].message.content or "{}"
    return _extract_json_object(content)


# ---------------------------------------------------------------------------
# 1. Baseline: 표준 표 양식(SUP-A) 기준 고정 정규표현식 스크립트
# ---------------------------------------------------------------------------
def run_baseline_regex(coa_text: str) -> Dict[str, Any]:
    """
    기존 자동화 한계 재현:
    - 고정 키워드(D50, BET, Fe, Moisture)와 표 구분자(|)에 의존
    - 단위 변환(nm->um, wt%->ppm, ppm->wt%) 미지원
    - 과거 이력 기반 SPC 드리프트 탐지 미지원
    """
    t0 = time.perf_counter()
    patterns = {
        "d50_um": r"D50\s*\(평균입도\)\s*\|\s*[^\|]+\|\s*([0-9.]+)",
        "bet_m2g": r"BET\s*\(비표면적\)\s*\|\s*[^\|]+\|\s*([0-9.]+)",
        "fe_ppm": r"Fe\s*\(철 불순물\)\s*\|\s*[^\|]+\|\s*([0-9.]+)",
        "moisture_wt": r"Moisture\s*\(수분함량\)\s*\|\s*[^\|]+\|\s*([0-9.]+)",
    }

    fallback_patterns = {
        "d50_um": r"(?:D50|d50)[^0-9\n]{0,20}([0-9.]+)",
        "bet_m2g": r"(?:BET)[^0-9\n]{0,25}([0-9.]+)",
        "fe_ppm": r"(?:Fe\s*(?:impurity|content|\(철))[^0-9\n]{0,25}([0-9.]+)",
        "moisture_wt": r"(?:Moisture)[^0-9\n]{0,25}([0-9.]+)",
    }

    extracted: Dict[str, Optional[float]] = {}
    for key, pat in patterns.items():
        m = re.search(pat, coa_text, flags=re.I)
        if not m:
            m = re.search(fallback_patterns[key], coa_text, flags=re.I)
        extracted[key] = float(m.group(1)) if m else None

    parse_success = all(v is not None for v in extracted.values())
    if not parse_success:
        return {
            "pipeline": "Baseline_Regex",
            "parse_success": False,
            "extracted_values": extracted,
            "predicted_status": "PARSE_ERROR",
            "latency_sec": round(time.perf_counter() - t0, 4),
            "trace": ["고정 정규식 파싱 실패: 비표준 양식 또는 용어(SSA/LOD/d(0.5)) 미인식"],
        }

    violations = []
    for metric, rule in SPEC_RULES.items():
        val = float(extracted[metric])  # type: ignore[arg-type]
        if val > rule["usl"] or (rule["lsl"] is not None and val < rule["lsl"]):
            violations.append(metric)

    status = "FAIL" if violations else "PASS"
    return {
        "pipeline": "Baseline_Regex",
        "parse_success": True,
        "extracted_values": extracted,
        "predicted_status": status,
        "latency_sec": round(time.perf_counter() - t0, 4),
        "trace": [
            f"정규식 추출 완료(단위 미환산): {extracted}",
            f"단건 규격 판정: {status} (이탈 항목: {violations or '없음'})",
        ],
    }


# ---------------------------------------------------------------------------
# 2. Agent v1: 단순 LLM 1회 호출 (파이썬 툴·자가검증·SPC DB 없음)
# ---------------------------------------------------------------------------
V1_SYSTEM_PROMPT = """당신은 세라믹 원료 분말 품질검사(IQC) 보조 AI입니다.
주어진 공급사 시험성적서(COA) 원문을 읽고 아래 사내 표준 규격과 비교하여 JSON만 출력하세요.

[사내 표준 규격]
- d50_um (평균 입도): 0.40 ~ 0.60 um
- bet_m2g (비표면적, SSA 포함): 5.50 ~ 7.50 m2/g
- fe_ppm (철 불순물 Fe): 100.0 ppm 이하
- moisture_wt (수분 함량, LOD/H2O 포함): 0.50 wt% 이하

반드시 아래 JSON 스키마로만 응답하세요:
{
  "d50_um": 숫자 또는 null,
  "bet_m2g": 숫자 또는 null,
  "fe_ppm": 숫자 또는 null,
  "moisture_wt": 숫자 또는 null,
  "predicted_status": "PASS" | "FAIL" | "SPC_WARNING",
  "reason": "판정 사유 한 줄 요약"
}
주의: 규격 범위 숫자가 아니라 실제 측정 결과 숫자를 읽으세요."""


def run_agent_v1_simple_llm(
    coa_text: str,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """단순 LLM 1회 프롬프트: 파이썬 계산기·검증 루프·과거 이력 DB 없음."""
    t0 = time.perf_counter()
    llm = client or get_llm_client()
    trace: List[str] = []

    try:
        data = _chat_json(
            llm,
            messages=[
                {"role": "system", "content": V1_SYSTEM_PROMPT},
                {"role": "user", "content": f"다음 COA를 분석하세요:\n\n{coa_text}"},
            ],
            model=model,
        )
        extracted = {
            k: (float(data[k]) if data.get(k) is not None else None)
            for k in SPEC_RULES
        }
        status = str(data.get("predicted_status", "PARSE_ERROR")).upper().strip()
        if status not in ("PASS", "FAIL", "SPC_WARNING"):
            status = "PARSE_ERROR"
        parse_success = all(v is not None for v in extracted.values())
        trace.append(f"LLM 1회 직접 추출/판정 완료: {extracted}")
        trace.append(f"판정: {status} | 사유: {data.get('reason', '')}")
    except Exception as exc:
        extracted = {k: None for k in SPEC_RULES}
        status = "PARSE_ERROR"
        parse_success = False
        trace.append(f"LLM 호출/파싱 오류: {exc}")

    return {
        "pipeline": "Agent_v1_Simple_LLM",
        "parse_success": parse_success,
        "extracted_values": extracted,
        "predicted_status": status,
        "latency_sec": round(time.perf_counter() - t0, 4),
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# 3. Agent v2: 툴 결합 + 물리 검증 자가수정 루프 + 누적 SPC 통계 에이전트
# ---------------------------------------------------------------------------
V2_EXTRACT_SYSTEM_PROMPT = """당신은 세라믹 원료 분말 시험성적서(COA)에서 원문 측정값과 원문 단위를 그대로 추출하는 전문 파싱 에이전트입니다.
절대 암산으로 단위를 바꾸지 마세요. 문서에 적힌 숫자와 단위 문자열을 있는 그대로 JSON으로 반환하면, 단위 환산과 통계 계산은 파이썬 툴이 수행합니다.

[추출 대상 4개 항목]
1. d50_um: 평균 입도 측정값 (표기 예: D50, Mean Particle Size, Median Dia., d(0.5)) -> D10/D90이나 규격 범위 숫자를 뽑지 마세요.
2. bet_m2g: 비표면적 측정값 (표기 예: BET, SSA, Specific Surface Area, N2-BET)
3. fe_ppm: 철 불순물 측정값 (표기 예: Fe, Iron impurity, Fe content) -> Na, Si, Al 수치를 뽑지 마세요.
4. moisture_wt: 수분 함량 측정값 (표기 예: Moisture, LOD, Loss on Drying, H2O content, Residual Moisture)

반드시 아래 JSON 형식으로만 응답하세요:
{
  "d50_um": {"value": 숫자, "unit": "원문 단위 문자열"},
  "bet_m2g": {"value": 숫자, "unit": "원문 단위 문자열"},
  "fe_ppm": {"value": 숫자, "unit": "원문 단위 문자열"},
  "moisture_wt": {"value": 숫자, "unit": "원문 단위 문자열"}
}"""


def run_agent_v2_self_correcting(
    coa_text: str,
    supplier_history_df: pd.DataFrame,
    lot_id: str = "LOT-DEMO",
    supplier_name: str = "Unknown Supplier",
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    max_retries: int = 2,
    generate_scar: bool = True,
) -> Dict[str, Any]:
    """
    제안 에이전트(v2):
    Step 1) LLM이 원문 측정값+단위 JSON 추출 (암산 금지)
    Step 2) Python 툴(validate_and_standardize_coa)이 단위 강제 환산 + 물리 경계 검증
    Step 3) 검증 실패 시 피드백(correction_hints)을 LLM에 전달해 최대 max_retries회 자가 수정
    Step 4) Python 툴(evaluate_lot_and_spc)이 단건 규격 + 공급사 누적 Cpk·SPC 드리프트 판정
    Step 5) 이상(FAIL/SPC_WARNING) 시 SCAR 초안 자동 생성
    """
    t0 = time.perf_counter()
    llm = client or get_llm_client()
    trace: List[str] = []

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": V2_EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"아래 COA 원문에서 4개 항목의 측정값과 원문 단위를 추출하세요:\n\n{coa_text}"},
    ]

    raw_extraction: Dict[str, Any] = {}
    val_result: Dict[str, Any] = {"valid": False, "standardized_values": {}, "issues": [], "correction_hints": []}
    retries_used = 0

    for attempt in range(max_retries + 1):
        try:
            raw_extraction = _chat_json(llm, messages=messages, model=model)
            trace.append(f"[시도 {attempt + 1}] LLM 원문 추출: {json.dumps(raw_extraction, ensure_ascii=False)}")
        except Exception as exc:
            trace.append(f"[시도 {attempt + 1}] JSON 파싱 실패: {exc}")
            messages.append({"role": "user", "content": "유효한 JSON 형식으로만 다시 응답하세요."})
            retries_used += 1
            continue

        val_result = validate_and_standardize_coa(raw_extraction)
        if val_result["valid"]:
            trace.append(f"[시도 {attempt + 1}] 파이썬 단위 표준화·물리 검증 통과: {val_result['conversion_log']}")
            break

        trace.append(
            f"[시도 {attempt + 1}] 물리 검증 실패 -> 자가 수정 요청: {val_result['issues']}"
        )
        if attempt < max_retries:
            retries_used += 1
            feedback_msg = (
                "이전 추출 결과를 파이썬 물리 검증 툴에 넣은 결과 아래 오류가 발견되었습니다.\n"
                f"- 오류: {val_result['issues']}\n"
                f"- 수정 힌트: {val_result['correction_hints']}\n"
                "성적서 원문을 다시 꼼꼼히 확인하고 수정된 JSON만 출력하세요."
            )
            messages.append({"role": "assistant", "content": json.dumps(raw_extraction, ensure_ascii=False)})
            messages.append({"role": "user", "content": feedback_msg})

    if not val_result["valid"]:
        return {
            "pipeline": "Agent_v2_SelfCorrect_SPC",
            "parse_success": False,
            "retries_used": retries_used,
            "raw_extraction": raw_extraction,
            "extracted_values": val_result["standardized_values"],
            "conversion_log": val_result.get("conversion_log", {}),
            "predicted_status": "PARSE_ERROR",
            "eval_result": None,
            "scar_draft": "",
            "latency_sec": round(time.perf_counter() - t0, 4),
            "trace": trace,
        }

    std_vals = {k: float(v) for k, v in val_result["standardized_values"].items()}
    eval_result = evaluate_lot_and_spc(std_vals, supplier_history_df)
    status = eval_result["final_status"]

    trace.append(
        f"[SPC 툴 판정] 최종 등급={status} | BET Cpk={eval_result['spc_metrics']['bet_m2g']['cpk']} | "
        f"D50 Cpk={eval_result['spc_metrics']['d50_um']['cpk']}"
    )
    if eval_result["spec_violations"]:
        trace.append(f"[규격 이탈 상세] {eval_result['spec_violations']}")
    if eval_result["spc_alerts"]:
        trace.append(f"[SPC 드리프트 경고] {eval_result['spc_alerts']}")

    scar_draft = (
        build_scar_draft(lot_id=lot_id, supplier_name=supplier_name, eval_result=eval_result)
        if generate_scar
        else ""
    )

    return {
        "pipeline": "Agent_v2_SelfCorrect_SPC",
        "parse_success": True,
        "retries_used": retries_used,
        "raw_extraction": raw_extraction,
        "extracted_values": std_vals,
        "conversion_log": val_result["conversion_log"],
        "predicted_status": status,
        "eval_result": eval_result,
        "scar_draft": scar_draft,
        "latency_sec": round(time.perf_counter() - t0, 4),
        "trace": trace,
    }


if __name__ == "__main__":
    gt = json.load(open("data/ground_truth.json", encoding="utf-8"))
    hist_df = pd.read_csv("data/history_seed.csv")
    sample = gt[17]  # COA_018
    text = open(sample["coa_path"], encoding="utf-8").read()

    sup_hist = hist_df[hist_df["supplier_id"] == sample["supplier_id"]].copy()
    prior_rows = [
        {"lot_id": r["lot_id"], "supplier_id": r["supplier_id"], **r["true_values"]}
        for r in gt
        if r["supplier_id"] == sample["supplier_id"] and r["coa_id"] < sample["coa_id"] and r["true_status"] != "FAIL"
    ]
    if prior_rows:
        sup_hist = pd.concat([sup_hist, pd.DataFrame(prior_rows)], ignore_index=True)

    print(f"=== 단일 샘플 테스트: {sample['coa_id']} ({sample['supplier_name']}) | 정답={sample['true_status']} ===")
    print("1) Baseline:", run_baseline_regex(text)["predicted_status"])
    v1 = run_agent_v1_simple_llm(text)
    print("2) Agent v1:", v1["predicted_status"], "| 값:", v1["extracted_values"])
    v2 = run_agent_v2_self_correcting(text, sup_hist, lot_id=sample["lot_id"], supplier_name=sample["supplier_name"])
    print("3) Agent v2:", v2["predicted_status"], "| 값:", v2["extracted_values"])
    print("\n[Agent v2 실행 로그]")
    for line in v2["trace"]:
        print(" ", line)
