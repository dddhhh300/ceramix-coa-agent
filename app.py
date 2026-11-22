"""
app.py
세라믹스넷 (CeramixNet) — Ceramix Materials㈜ 사내 품질포털 & AI IQC·SPC 에이전트
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from agent_coa import (
    run_agent_v1_simple_llm,
    run_agent_v2_self_correcting,
    run_baseline_regex,
)
from coa_tools import SPEC_RULES

load_dotenv()

st.set_page_config(
    page_title="세라믹스넷 · Ceramix Materials㈜",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 인트라넷 스타일 (밝은/어두운 테마 모두 가독성 유지되도록 명시적 배경·글자색 지정)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .cm-topbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.7rem 1.25rem;
        background-color: #111827;
        color: #F9FAFB;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #1F2937;
    }
    .cm-brand {
        font-size: 1.08rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.01em;
    }
    .cm-brand small {
        font-size: 0.8rem;
        color: #9CA3AF;
        font-weight: 500;
        margin-left: 0.5rem;
    }
    .cm-user-pill {
        font-size: 0.82rem;
        color: #E5E7EB;
        background-color: #1F2937;
        border: 1px solid #374151;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
    }
    .cm-hero {
        background: linear-gradient(120deg, #1E293B 0%, #0F4C81 65%, #0369A1 100%);
        color: #FFFFFF;
        padding: 1.55rem 1.8rem;
        border-radius: 12px;
        margin-bottom: 1.15rem;
    }
    .cm-hero-kicker {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        color: #BAE6FD;
        margin-bottom: 0.35rem;
    }
    .cm-hero h2 {
        margin: 0 0 0.4rem 0;
        font-size: 1.55rem;
        font-weight: 800;
        color: #FFFFFF;
    }
    .cm-hero p {
        margin: 0;
        font-size: 0.94rem;
        color: #E2E8F0;
        line-height: 1.55;
    }
    .cm-card {
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    .cm-card-accent {
        background-color: #EFF6FF;
        color: #0F172A;
        border: 1px solid #93C5FD;
        border-left: 5px solid #1D4ED8;
        border-radius: 10px;
        padding: 1.05rem 1.2rem;
        margin-bottom: 0.85rem;
    }
    .cm-tag {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background-color: #E2E8F0;
        color: #1E293B;
        margin-right: 6px;
    }
    .cm-tag-urgent {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background-color: #FEE2E2;
        color: #991B1B;
        margin-right: 6px;
    }
    .cm-tag-blue {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        background-color: #DBEAFE;
        color: #1E40AF;
        margin-right: 6px;
    }
    .cm-mail-box {
        background-color: #FFFFFF;
        color: #0F172A;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        line-height: 1.7;
    }
    .cm-mail-meta {
        font-size: 0.86rem;
        color: #475569;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
    }
    .badge-pass {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #DCFCE7; color: #166534; font-weight: 700; font-size: 0.88rem;
    }
    .badge-warn {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #FFEDD5; color: #9A3412; font-weight: 700; font-size: 0.88rem;
    }
    .badge-fail {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #FEE2E2; color: #991B1B; font-weight: 700; font-size: 0.88rem;
    }
    .badge-err {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #E2E8F0; color: #334155; font-weight: 700; font-size: 0.88rem;
    }
    .cm-footer {
        margin-top: 2rem;
        padding: 1rem 1.3rem;
        background-color: #F8FAFC;
        color: #475569;
        border-top: 1px solid #CBD5E1;
        border-radius: 8px;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_badge_html(status: str) -> str:
    s = str(status).upper()
    if s == "PASS":
        return '<span class="badge-pass">PASS · 정상 입고 승인</span>'
    if s == "SPC_WARNING":
        return '<span class="badge-warn">SPC_WARNING · 공정 드리프트 선제 경고</span>'
    if s == "FAIL":
        return '<span class="badge-fail">FAIL · 규격 이탈 (입고 보류)</span>'
    return f'<span class="badge-err">{s}</span>'


# ---------------------------------------------------------------------------
# 데이터 로드 및 유틸리티
# ---------------------------------------------------------------------------
@st.cache_data
def load_project_artifacts() -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.DataFrame, List[Dict[str, Any]]]:
    history_df = pd.read_csv("data/history_seed.csv")
    with open("data/ground_truth.json", encoding="utf-8") as f:
        gt_records = json.load(f)
    summary_df = pd.read_csv("results/benchmark_summary.csv")
    with open("results/detailed_predictions.json", encoding="utf-8") as f:
        detailed = json.load(f)
    return history_df, gt_records, summary_df, detailed


def get_supplier_history_before(
    supplier_id: str,
    target_coa_id: Optional[str],
    history_seed_df: pd.DataFrame,
    gt_records: List[Dict[str, Any]],
) -> pd.DataFrame:
    sup_hist = history_seed_df[history_seed_df["supplier_id"] == supplier_id].copy()
    prior_rows = [
        {
            "lot_id": r["lot_id"],
            "supplier_id": r["supplier_id"],
            "supplier_name": r["supplier_name"],
            "date": r["date"],
            **r["true_values"],
            "status": r["true_status"],
        }
        for r in gt_records
        if r["supplier_id"] == supplier_id
        and (target_coa_id is None or r["coa_id"] < target_coa_id)
        and r["true_status"] != "FAIL"
    ]
    if prior_rows:
        sup_hist = pd.concat([sup_hist, pd.DataFrame(prior_rows)], ignore_index=True)
    return sup_hist


def resolve_api_config(
    sidebar_key: str, sidebar_base_url: str, sidebar_model: str
) -> Tuple[str, str, str]:
    def _secret(key: str) -> str:
        try:
            return str(st.secrets.get(key, "")).strip()
        except Exception:
            return ""

    api_key = sidebar_key.strip() or _secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
    base_url = sidebar_base_url.strip() or _secret("OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL", "").strip()
    model = sidebar_model.strip() or _secret("OPENAI_MODEL") or os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    return api_key, base_url, (model or "gpt-4o-mini")


def plot_single_lot_spc(
    sup_hist: pd.DataFrame,
    current_lot_id: str,
    current_val: float,
    metric: str,
    supplier_id: str,
) -> plt.Figure:
    rule = SPEC_RULES[metric]
    hist_tail = sup_hist.tail(19)
    labels = list(hist_tail["lot_id"]) + [current_lot_id]
    values = list(hist_tail[metric].astype(float)) + [float(current_val)]
    x = np.arange(1, len(values) + 1)

    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=140)
    ax.plot(x[:-1], values[:-1], color="#64748B", marker="o", linewidth=1.6, markersize=4.5, label="Past Lots")
    ax.plot(x[-2:], values[-2:], color="#2563EB", linewidth=2.0)

    is_fail = current_val > rule["usl"] or (rule["lsl"] is not None and current_val < rule["lsl"])
    cur_color = "#DC2626" if is_fail else "#2563EB"
    ax.scatter([x[-1]], [values[-1]], color=cur_color, s=95, zorder=5, label=f"Current ({current_lot_id})")

    ax.axhline(rule["usl"], color="#DC2626", linestyle="--", linewidth=1.4, label=f"USL ({rule['usl']})")
    if rule["lsl"] is not None:
        ax.axhline(rule["lsl"], color="#DC2626", linestyle="--", linewidth=1.4, label=f"LSL ({rule['lsl']})")
    ax.axhline(rule["target"], color="#0F766E", linestyle=":", linewidth=1.2, label=f"Target ({rule['target']})")

    ax.set_title(f"SPC Control Chart [{supplier_id}] - {metric} ({rule['unit']})", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(f"{metric} ({rule['unit']})", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("LOT-2026-", "L") for l in labels], rotation=45, ha="right", fontsize=7.5)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=7.5)
    plt.tight_layout()
    return fig


def build_v1_error_audit(detailed_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Agent v1(단순 LLM)이 실제로 틀린 건만 자동 추출해 'AI가 틀린 것' 감사표 구성."""
    rows = []
    for r in detailed_results:
        v1 = r["agent_v1"]
        v2 = r["agent_v2"]
        if v1["status_exact"] and v1["value_exact"]:
            continue

        diffs = []
        for m, t_val in r["true_values"].items():
            p_val = v1["extracted_values"].get(m)
            if p_val is None or abs(float(p_val) - float(t_val)) > max(abs(float(t_val)) * 0.02, 1e-3):
                diffs.append(f"{m}: 실제 {t_val} -> v1 {p_val}")

        if diffs:
            err_type = "단위 환산 / 수치 추출 오류"
            err_detail = " | ".join(diffs)
            fix_how = "LLM 암산 금지 + Python 단위 표준화·물리 경계 검증 툴로 정확 환산"
        elif r["true_status"] == "SPC_WARNING":
            err_type = "누적 SPC 드리프트 미탐지"
            err_detail = f"단건 규격 내 수치라 PASS 오판 (실제: 7연속/추세 편향)"
            fix_how = "공급사 과거 이력 DB 결합 + Python Cpk·Run7/Trend6 통계 판정"
        else:
            err_type = "단건 규격 판정 오류"
            err_detail = f"정답 {r['true_status']} -> v1 {v1['predicted_status']}"
            fix_how = "Python 규격 비교 툴(USL/LSL) 분리 적용"

        rows.append(
            {
                "COA ID": r["coa_id"],
                "공급사": r["supplier_id"],
                "오류 유형": err_type,
                "Agent v1 오판 내용": err_detail,
                "실제 정답": r["true_status"],
                "Agent v1 판정": v1["predicted_status"],
                "Agent v2 판정": v2["predicted_status"],
                "개선 및 검증 방법": fix_how,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 사내 포털 콘텐츠 데이터 (Ceramix Materials㈜)
# ---------------------------------------------------------------------------
MAILS = [
    {
        "id": "m-iqc-task",
        "unread": True,
        "flag": True,
        "from": "박준혁 <jh.park@ceramix-mat.co.kr> · 품질보증팀장",
        "to": "품질보증팀 IQC 담당 엔지니어",
        "cc": "이서연 (소결공정팀장), 김도현 (구매자재팀 과장)",
        "date": "2026-11-16 (월) 오전 08:35",
        "subject": "[긴급 과제] 해외 4개국 분말 성적서(COA) 수입검사 자동화 및 SPC 드리프트 검증 요청",
        "attach": "data/coa_samples (40건), data/history_seed.csv, results/benchmark_summary.csv",
        "body": """수입검사(IQC) 업무에 합류하자마자 급한 현안을 맡깁니다.

최근 연료전지 전해질 기판(CMX-SOFC-E1) 라인에서 **슬러리 점도 급상승과 소결 수축률 편차**가 잇따라 발생했습니다.
원인을 추적해 보니 주원료인 고순도 산화물 분말(**CMX-YSZ-05**)을 한국·일본·중국·독일 4개 공급사에서 들여오는 과정에서 두 가지 품질 구멍이 뚫려 있었습니다.

1. **비정형 COA 단위 혼선:** 일본(SUP-B)·중국(SUP-C)·독일(SUP-D) 공급사가 `nm`/`μm`, `wt%`/`ppm`, `SSA`/`LOD`/`d(0.5)`를 제각각 표기합니다. 기존 정규식 스크립트는 국내 표준 양식(SUP-A)만 읽어 해외 30건이 전부 수기 입력으로 넘어오고, 지난달 작게 테스트한 **단순 AI 프롬프트(v1)도 `wt%→ppm` 암산 오류로 불량 로트를 놓쳤습니다.**
2. **규격 내 '조용한 드리프트' 누락:** 단건 성적서 수치만 보고 합격 도장을 찍다 보니, 일본 SUP-B의 비표면적(BET)과 독일 SUP-D의 입도(D50)가 7로트 연속 한쪽으로 쏠리는 공정 이상을 아무도 못 잡았습니다.

**[요청 사항]**
1. 4개국 비정형 성적서 40건과 과거 이력 DB를 기준으로, **단위 환산·물리 검증·누적 SPC(Cpk 및 7연속 추세) 판정·SCAR 초안 작성**까지 수행하는 AI 에이전트(v2)를 구축해 주세요.
2. 기존 정규식(Baseline), 단순 LLM(v1), 개선 에이전트(v2)의 **40건 정량 비교표**와 **v1이 실제로 어디서 틀렸는지(오류 감사표)**를 함께 정리해 주세요.
3. 이번 COA 샘플 데이터는 정보보안팀 검토를 거쳐 **사내 승인 API 사용 허가**를 받았습니다. 다만 최종 판정 수치는 반드시 파이썬 결정론적 툴로 검증되도록 설계하세요.

세라믹 원료 물성 기준은 포털의 **'교육 자료 · 규격서'** 메뉴를 참고하고, 결과물은 이번 주 품질혁신 회의에서 시연할 수 있게 포털에 올려 두세요.

박준혁 드림
Ceramix Materials㈜ 품질보증팀장 · 내선 4210""",
    },
    {
        "id": "m-sinter",
        "unread": False,
        "flag": True,
        "from": "이서연 <sy.lee@ceramix-mat.co.kr> · 소결공정팀장",
        "to": "품질보증팀 IQC 담당 엔지니어",
        "cc": "박준혁 (품질보증팀장)",
        "date": "2026-11-17 (화) 오후 02:10",
        "subject": "[공정 협조] SUP-B 분말 입고 로트 슬러리 점도 상승 건 공유",
        "attach": "results/spc_drift_sup_b_bet.png",
        "body": """품질보증팀 담당자님, 소결공정팀 이서연입니다.

지난주 투입한 일본 Nippon Fine Powder(SUP-B) 분말 로트에서 테이프 캐스팅 슬러리 점도가 평소보다 18% 높게 나와 기포 제거 시간이 길어졌습니다.
수입검사 대장을 보니 단건 성적서 상 BET 비표면적은 7.5 m²/g 이하로 '합격' 처리되어 있었는데, 최근 7~8개 로트를 이어 붙여 보니 BET가 6.46에서 7.31 m²/g까지 계속 우상향하고 있었습니다.

규격 상한선(USL)을 넘기 전에 **연속 추세(SPC 드리프트)가 보이면 선제 경고(`SPC_WARNING`)와 공급사 SCAR 초안**이 바로 뜨도록 AI 수입검사 시스템에 꼭 반영 부탁드립니다.

이서연 드림 · 소결공정팀 (내선 4310)""",
    },
    {
        "id": "m-it-sec",
        "unread": False,
        "flag": False,
        "from": "정보보안·DX팀 <dx-sec@ceramix-mat.co.kr>",
        "to": "품질보증팀",
        "cc": "",
        "date": "2026-11-15 (일) 오후 05:20",
        "subject": "[보안 승인] CMX-YSZ-05 합성 COA 데이터 및 OpenAI 호환 API 연동 승인 안내",
        "attach": "",
        "body": """품질보증팀의 '비정형 COA 자동 검증 에이전트' 과제에 대해 정보보안 검토를 완료했습니다.

- **승인 범위:** 비식별화/합성 처리된 공급사 4곳 시험성적서 40건 및 과거 이력 시드 데이터
- **보안 수칙:** API 키는 서버 환경변수(`.env` / Secrets)로만 관리하며, 외부 공개 저장소(GitHub)에 키 파일이 포함되지 않도록 `.gitignore`를 적용할 것
- **감사 요구:** LLM이 임의로 단위 암산이나 합부 판정을 내리지 않도록 환산·통계 로직은 검증 가능한 Python 코드로 분리 기록할 것

정보보안·DX팀 (내선 4100)""",
    },
    {
        "id": "m-ga",
        "unread": False,
        "flag": False,
        "from": "경영지원팀 <ga@ceramix-mat.co.kr>",
        "to": "전 임직원",
        "cc": "",
        "date": "2026-11-13 (금) 오전 09:00",
        "subject": "[공지] 11월 화순 제1공장 소성동 안전점검 및 셔틀버스 운행 안내",
        "attach": "",
        "body": """1. **소성동·분말창고 정기 안전점검:** 11월 25일(수) 14:00 ~ 16:00 (방진마스크·안전화 필수 착용)
2. **광주 첨단 R&D 본사 ↔ 화순 제1공장 셔틀:** 오전 07:40 출발 / 오후 18:10 복귀 정상 운행
3. **구내식당 운영:** 조식 07:30~08:20 · 중식 11:40~13:00 · 석식 17:40~18:30

경영지원팀 (내선 4010)""",
    },
]

NOTICES = [
    {"tag": "품질", "urgent": True, "title": "CMX-YSZ-05 해외 공급사 AI 수입검사 시스템(Agent v2) 시범 가동", "date": "2026-11-22", "dept": "품질보증팀"},
    {"tag": "공정", "urgent": False, "title": "SUP-B(BET)·SUP-D(D50) 연속 드리프트 로트 선제 모니터링 적용", "date": "2026-11-20", "dept": "소결공정팀"},
    {"tag": "보안", "urgent": False, "title": "사내 AI 도구 활용 시 외부 API 키 비노출 및 계산 검증 수칙 준수", "date": "2026-11-15", "dept": "정보보안·DX팀"},
    {"tag": "안전", "urgent": False, "title": "분말 계량실·소성동 출입 시 방진마스크 및 안전화 착용 의무", "date": "2026-11-13", "dept": "환경안전팀"},
]

SCHEDULE = [
    {"date": "11-16 (월)", "title": "IQC 자동화 과제 착수 · 공급사 4곳 성적서 수집", "where": "품질동"},
    {"date": "11-19 (목)", "title": "Agent v1(단순 LLM) 오류 감사 및 물리 검증 루프 설계", "where": "R&D 센터"},
    {"date": "11-22 (일)", "title": "40건 벤치마크 확정 · 세라믹스넷 포털 배포 및 최종 제출", "where": "품질보증팀"},
    {"date": "11-25 (수)", "title": "화순 제1공장 소성동·분말창고 정기 안전점검", "where": "화순 제1공장"},
    {"date": "11-27 (금)", "title": "SUP-B · SUP-D 공급사 SCAR 회신 검토 회의", "where": "본관 2층"},
]

PEOPLE = [
    {"이름": "박준혁", "부서": "품질보증팀", "직책": "팀장", "내선": "4210", "이메일": "jh.park@ceramix-mat.co.kr", "위치": "품질동 2층", "담당 업무": "수입검사(IQC) · 공급사 품질(SQE) 총괄"},
    {"이름": "장도영", "부서": "품질보증팀", "직책": "신입 엔지니어", "내선": "4215", "이메일": "dy.jang@ceramix-mat.co.kr", "위치": "품질동 2층", "담당 업무": "원료 COA 검증 · AI SPC 자동화 구축"},
    {"이름": "이서연", "부서": "소결공정팀", "직책": "팀장", "내선": "4310", "이메일": "sy.lee@ceramix-mat.co.kr", "위치": "화순 제1공장", "담당 업무": "슬러리 분산 · 테이프 캐스팅 · 고온 소결"},
    {"이름": "최민재", "부서": "세라믹소재연구소", "직책": "책임연구원", "내선": "4410", "이메일": "mj.choi@ceramix-mat.co.kr", "위치": "첨단 R&D 3층", "담당 업무": "YSZ·BCZYYb 분말 계면·소결 거동 분석"},
    {"이름": "김도현", "부서": "구매자재팀", "직책": "과장", "내선": "4510", "이메일": "dh.kim@ceramix-mat.co.kr", "위치": "본관 1층", "담당 업무": "한·일·중·독 4개국 원료 소싱 및 SCAR 발송"},
    {"이름": "정유진", "부서": "정보보안·DX팀", "직책": "선임", "내선": "4105", "이메일": "yj.jung@ceramix-mat.co.kr", "위치": "본관 3층", "담당 업무": "사내 포털(세라믹스넷) · AI API 보안 검토"},
]

# ---------------------------------------------------------------------------
# 데이터 로드 & 페이지 상태 초기화
# ---------------------------------------------------------------------------
history_seed_df, gt_records, summary_df, detailed_results = load_project_artifacts()
detailed_by_id = {r["coa_id"]: r for r in detailed_results}
v1_audit_df = build_v1_error_audit(detailed_results)

row_base = summary_df[summary_df["Method"] == "baseline"].iloc[0]
row_v1 = summary_df[summary_df["Method"] == "agent_v1"].iloc[0]
row_v2 = summary_df[summary_df["Method"] == "agent_v2"].iloc[0]

PAGES = [
    "홈",
    "업무 메일",
    "AI 수입검사 시스템",
    "검증 리포트 · AI가 틀린 것",
    "자료실 (COA · DB)",
    "교육 자료 · 규격서",
    "회사 소개 · 사람 찾기",
]

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "홈"


def go_to(page_name: str) -> None:
    st.session_state["nav_page"] = page_name


# ---------------------------------------------------------------------------
# 좌측 사내 인트라넷 네비게이션 사이드바
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🏭 세라믹스넷")
    st.caption("Ceramix Materials㈜ 사내 업무 포털")
    st.divider()

    selected_page = st.radio(
        "사내 메뉴",
        PAGES,
        key="nav_page",
        label_visibility="collapsed",
    )

    st.divider()
    with st.expander("⚙️ 실시간 AI API 설정 (선택)", expanded=False):
        st.caption("기본 '사전 검증 결과 즉시 재생' 모드는 키 없이 40건 전체 시연 가능합니다.")
        sb_api_key = st.text_input("OpenAI 호환 API Key", type="password", placeholder="미입력 시 서버/로컬 키 사용")
        sb_model = st.text_input("Model", value="", placeholder="기본: gpt-4o-mini")
        sb_base_url = st.text_input("Base URL (선택)", value="", placeholder="공식 OpenAI는 비워둠")

    resolved_key, resolved_url, resolved_model = resolve_api_config(sb_api_key, sb_base_url, sb_model)
    if resolved_key:
        st.success(f"실시간 API 연동됨 ({resolved_model})")
    else:
        st.info("사전 검증 재생 모드 활성 (키 불필요)")

    st.divider()
    st.caption("Ceramix Materials㈜ · 정보보안·DX팀 내선 4100")

# ---------------------------------------------------------------------------
# 상단 공통 인트라넷 헤더 바
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cm-topbar">
      <div class="cm-brand">
        세라믹스넷 <small>Ceramix Materials㈜ · 고순도 산화물 세라믹 제조</small>
      </div>
      <div class="cm-user-pill">
        2026-11-22 &nbsp;|&nbsp; <strong>장도영</strong> (품질보증팀 · 사번 261101)
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ===========================================================================
# PAGE 1: 홈 (사내 포털 대시보드)
# ===========================================================================
if selected_page == "홈":
    st.markdown(
        """
        <div class="cm-hero">
          <div class="cm-hero-kicker">CERAMIX MATERIALS INTRANET · QUALITY FIRST</div>
          <h2>분말을 읽는 기술, 무결점 소결의 시작</h2>
          <p>
            Ceramix Materials㈜는 연료전지 전해질 기판과 정밀 산화물 세라믹 부품을 제조합니다.
            4개국 원료 공급사 시험성적서(COA)의 자동 표준화, 물리 타당성 검증, 누적 SPC 드리프트 감지를 통해
            원료 입고 단계부터 공정 신뢰성을 관리합니다.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 현재 진행 과제 요약 배너 + 빠른 이동 버튼
    st.markdown(
        f"""
        <div class="cm-card-accent">
          <span class="cm-tag-urgent">진행 중 핵심 과제</span>
          <strong>[품질보증팀] 해외 4개국 분말 COA 자동 검증 및 SPC 드리프트 탐지 에이전트(v2) 구축</strong><br/>
          <span style="font-size:0.9rem; color:#334155;">
            40건 비정형 COA 벤치마크 확정 — 종합 판정 정확도 <strong>{row_base['Overall_Decision_Accuracy_pct']:.1f}% (정규식) → {row_v1['Overall_Decision_Accuracy_pct']:.1f}% (단순 LLM) → {row_v2['Overall_Decision_Accuracy_pct']:.1f}% (Agent v2)</strong>
            &nbsp;|&nbsp; 누적 SPC 드리프트 탐지율 <strong>{row_v2['SPC_Drift_Recall_pct']:.1f}%</strong>
            &nbsp;|&nbsp; 건당 <strong>{row_v2['Avg_Latency_sec']:.2f}초</strong>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    b1, b2, b3, b4 = st.columns(4)
    b1.button("📩 팀장 업무 지시 메일 열기", on_click=go_to, args=("업무 메일",), use_container_width=True)
    b2.button("🤖 AI 수입검사 시스템 실행", type="primary", on_click=go_to, args=("AI 수입검사 시스템",), use_container_width=True)
    b3.button("🔍 AI가 틀린 것 · 40건 검증표", on_click=go_to, args=("검증 리포트 · AI가 틀린 것",), use_container_width=True)
    b4.button("📁 자료실 (COA 40건 · DB)", on_click=go_to, args=("자료실 (COA · DB)",), use_container_width=True)

    st.write("")
    col_left, col_right = st.columns([1.35, 1])

    with col_left:
        st.markdown("#### 📢 사내 공지사항")
        for n in NOTICES:
            tag_cls = "cm-tag-urgent" if n["urgent"] else "cm-tag"
            st.markdown(
                f"""
                <div class="cm-card" style="padding:0.75rem 1rem; margin-bottom:0.55rem;">
                  <span class="{tag_cls}">{n['tag']}</span>
                  <strong>{n['title']}</strong><br/>
                  <span style="font-size:0.8rem; color:#64748B;">{n['dept']} · {n['date']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### 📰 현장 소식 (Ceramix Story)")
        n1, n2 = st.columns(2)
        with n1:
            st.markdown(
                """
                <div class="cm-card">
                  <span class="cm-tag-blue">현장 DX</span>
                  <strong>단위 함정과 '조용한 불량'을 잡다</strong>
                  <p style="font-size:0.86rem; color:#334155; margin-top:0.45rem;">
                    일본·중국·독일 공급사의 nm/μm·wt%/ppm 혼선과 단건 합격 뒤에 숨은 7연속 BET·D50 편향을
                    LLM 파싱과 Python 통계 툴 분리 구조로 해결했습니다.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with n2:
            st.markdown(
                """
                <div class="cm-card">
                  <span class="cm-tag-blue">주력 제품</span>
                  <strong>CMX-YSZ-05 분말과 전해질 기판</strong>
                  <p style="font-size:0.86rem; color:#334155; margin-top:0.45rem;">
                    고체산화물 연료전지(SOFC) 전해질의 치밀도는 원료 입도(D50 0.40~0.60 μm)와
                    비표면적(BET 5.50~7.50 m²/g)의 균일성에서 결정됩니다.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_right:
        st.markdown("#### 📅 주요 일정")
        sched_df = pd.DataFrame(SCHEDULE).rename(columns={"date": "날짜", "title": "일정", "where": "장소"})
        st.dataframe(sched_df, hide_index=True)

        st.markdown("#### 📊 핵심 원료(CMX-YSZ-05) 품질 현황")
        k1, k2 = st.columns(2)
        k1.metric("검증 완료 COA", "40건", "4개국 공급사")
        k2.metric("규격 이탈(FAIL) 차단", "10 / 10건", "100.0% 탐지")
        k3, k4 = st.columns(2)
        k3.metric("SPC 드리프트 선제 경고", "8 / 8건", "SUP-B · SUP-D")
        k4.metric("Agent v1 오류 적발", f"{len(v1_audit_df)}건", "v2에서 전량 해결")


# ===========================================================================
# PAGE 2: 업무 메일 (불뫼넷 메일함 스타일)
# ===========================================================================
elif selected_page == "업무 메일":
    st.subheader("📩 사내 메일함 (받은 편지함 4건)")
    col_list, col_body = st.columns([1, 1.65])

    with col_list:
        mail_idx = st.radio(
            "메일 목록",
            range(len(MAILS)),
            format_func=lambda i: f"{'🔴 ' if MAILS[i]['unread'] else '⚪ '}{MAILS[i]['subject']}\n   ({MAILS[i]['date']} · {MAILS[i]['from'].split('·')[-1].strip()})",
            label_visibility="collapsed",
        )

    m = MAILS[mail_idx]
    with col_body:
        st.markdown(
            f"""
            <div class="cm-mail-box">
              <h4 style="margin:0 0 0.6rem 0; color:#0F172A;">{m['subject']}</h4>
              <div class="cm-mail-meta">
                <strong>보낸 사람:</strong> {m['from']}<br/>
                <strong>받는 사람:</strong> {m['to']}<br/>
                {f"<strong>참조:</strong> {m['cc']}<br/>" if m['cc'] else ""}
                <strong>날짜:</strong> {m['date']}<br/>
                {f"<strong>첨부 자료:</strong> <code>{m['attach']}</code>" if m['attach'] else ""}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(m["body"])
        if m["id"] == "m-iqc-task":
            st.divider()
            c_btn1, c_btn2 = st.columns(2)
            c_btn1.button("🤖 구축된 AI 수입검사 시스템(v2) 바로 실행", type="primary", on_click=go_to, args=("AI 수입검사 시스템",), use_container_width=True)
            c_btn2.button("🔍 40건 벤치마크 · AI가 틀린 것 보기", on_click=go_to, args=("검증 리포트 · AI가 틀린 것",), use_container_width=True)


# ===========================================================================
# PAGE 3: AI 수입검사 시스템 (라이브 데모)
# ===========================================================================
elif selected_page == "AI 수입검사 시스템":
    st.subheader("🤖 Smart IQC — 비정형 COA 검증 · SPC 드리프트 탐지 · SCAR 자동 작성")

    mode = st.radio(
        "실행 모드 선택",
        [
            "사전 검증 결과 즉시 재생 (40건 전체 · API 키 불필요 · 빠른 시연 권장)",
            "실시간 AI 에이전트 직접 실행 (원문 수정/신규 입력 · OpenAI 호환 API 호출)",
        ],
        horizontal=True,
    )

    highlight_presets = {
        "COA_018 (일본 SUP-B · nm/wt% 단위 함정 + BET 7연속 SPC 드리프트)": "COA_018",
        "COA_024 (중국 SUP-C · 이메일 본문 + Fe wt% 단위 함정 -> FAIL)": "COA_024",
        "COA_026 (중국 SUP-C · 이메일 본문 + 수분 ppm 단위 함정 -> FAIL)": "COA_026",
        "COA_038 (독일 SUP-D · 파이프 표 + D50 7연속 하향 드리프트)": "COA_038",
        "COA_003 (한국 SUP-A · 표준 표 + Fe 규격 초과 -> FAIL)": "COA_003",
        "COA_001 (한국 SUP-A · 정상 입고 -> PASS)": "COA_001",
    }

    col_sel1, col_sel2 = st.columns([1.3, 1])
    with col_sel1:
        preset_choice = st.selectbox("대표 시연 시나리오 빠른 선택", ["직접 전체 40건 중 선택"] + list(highlight_presets.keys()))
    with col_sel2:
        all_ids = [r["coa_id"] for r in detailed_results]
        default_idx = all_ids.index(highlight_presets[preset_choice]) if preset_choice != "직접 전체 40건 중 선택" else 17
        selected_coa_id = st.selectbox(
            "40건 COA 샘플 ID 선택",
            all_ids,
            index=default_idx,
            format_func=lambda cid: f"{cid} | {detailed_by_id[cid]['supplier_id']} | 정답: {detailed_by_id[cid]['true_status']}",
        )

    rec = detailed_by_id[selected_coa_id]
    raw_text = open(rec["coa_path"], encoding="utf-8").read()
    sup_hist = get_supplier_history_before(rec["supplier_id"], rec["coa_id"], history_seed_df, gt_records)

    if mode.startswith("사전"):
        left_col, right_col = st.columns([1.05, 1.25])

        with left_col:
            st.markdown(f"#### 공급사 원본 COA 문서 (`{rec['coa_id']}` · {rec['supplier_name']})")
            st.caption(f"시나리오: `{rec['scenario_tag']}` | 포함 함정: `{', '.join(rec['traps']) if rec['traps'] else '없음(표준)'}`")
            st.code(raw_text, language="text")

            st.markdown("#### 동일 문서 3종 파이프라인 판정 비교")
            comp_df = pd.DataFrame(
                [
                    {
                        "방식": "정답 (Ground Truth)",
                        "판정": rec["true_status"],
                        "D50 (μm)": rec["true_values"]["d50_um"],
                        "BET (m²/g)": rec["true_values"]["bet_m2g"],
                        "Fe (ppm)": rec["true_values"]["fe_ppm"],
                        "수분 (wt%)": rec["true_values"]["moisture_wt"],
                    },
                    {
                        "방식": "Baseline (정규식)",
                        "판정": rec["baseline"]["predicted_status"],
                        "D50 (μm)": rec["baseline"]["extracted_values"]["d50_um"],
                        "BET (m²/g)": rec["baseline"]["extracted_values"]["bet_m2g"],
                        "Fe (ppm)": rec["baseline"]["extracted_values"]["fe_ppm"],
                        "수분 (wt%)": rec["baseline"]["extracted_values"]["moisture_wt"],
                    },
                    {
                        "방식": "Agent v1 (단순 LLM)",
                        "판정": rec["agent_v1"]["predicted_status"],
                        "D50 (μm)": rec["agent_v1"]["extracted_values"]["d50_um"],
                        "BET (m²/g)": rec["agent_v1"]["extracted_values"]["bet_m2g"],
                        "Fe (ppm)": rec["agent_v1"]["extracted_values"]["fe_ppm"],
                        "수분 (wt%)": rec["agent_v1"]["extracted_values"]["moisture_wt"],
                    },
                    {
                        "방식": "Agent v2 (사내 AI 시스템)",
                        "판정": rec["agent_v2"]["predicted_status"],
                        "D50 (μm)": rec["agent_v2"]["extracted_values"]["d50_um"],
                        "BET (m²/g)": rec["agent_v2"]["extracted_values"]["bet_m2g"],
                        "Fe (ppm)": rec["agent_v2"]["extracted_values"]["fe_ppm"],
                        "수분 (wt%)": rec["agent_v2"]["extracted_values"]["moisture_wt"],
                    },
                ]
            )
            st.dataframe(comp_df, hide_index=True)

        with right_col:
            v2 = rec["agent_v2"]
            st.markdown("#### Agent v2 분석 결과 및 공정 판정")
            st.markdown(
                f"최종 판정: {status_badge_html(v2['predicted_status'])} &nbsp;&nbsp; "
                f"| 처리 시간: **{v2['latency_sec']:.2f}초** | 자가 수정: **{v2['retries_used']}회**",
                unsafe_allow_html=True,
            )

            eval_res = v2.get("eval_result")
            if eval_res:
                st.info(eval_res["decision_summary"])
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("D50 (μm)", f"{v2['extracted_values']['d50_um']:.3f}", f"Cpk {eval_res['spc_metrics']['d50_um']['cpk']:.2f}")
                m2.metric("BET (m²/g)", f"{v2['extracted_values']['bet_m2g']:.2f}", f"Cpk {eval_res['spc_metrics']['bet_m2g']['cpk']:.2f}")
                m3.metric("Fe (ppm)", f"{v2['extracted_values']['fe_ppm']:.1f}", f"Cpk {eval_res['spc_metrics']['fe_ppm']['cpk']:.2f}")
                m4.metric("Moisture (wt%)", f"{v2['extracted_values']['moisture_wt']:.2f}", f"Cpk {eval_res['spc_metrics']['moisture_wt']['cpk']:.2f}")

                default_metric = "bet_m2g"
                if eval_res["spec_violations"]:
                    default_metric = eval_res["spec_violations"][0]["metric"]
                elif eval_res["spc_alerts"]:
                    default_metric = eval_res["spc_alerts"][0]["metric"]

                chart_metric = st.selectbox(
                    "SPC 관리도 표시 항목",
                    list(SPEC_RULES.keys()),
                    index=list(SPEC_RULES.keys()).index(default_metric),
                    format_func=lambda k: SPEC_RULES[k]["label"],
                )
                fig = plot_single_lot_spc(
                    sup_hist,
                    rec["lot_id"],
                    float(v2["extracted_values"][chart_metric]),
                    chart_metric,
                    rec["supplier_id"],
                )
                st.pyplot(fig, clear_figure=True)

            with st.expander("단위 표준화 및 에이전트 단계별 실행 로그 (Trace)", expanded=True):
                for step_line in v2["trace"]:
                    st.markdown(f"- `{step_line}`")

            if v2.get("scar_draft"):
                with st.expander("자동 생성된 공급사 시정조치요구서 (SCAR) 초안", expanded=(v2["predicted_status"] != "PASS")):
                    st.text(v2["scar_draft"])
                    st.download_button(
                        label="SCAR 문서 다운로드 (.txt)",
                        data=v2["scar_draft"].encode("utf-8"),
                        file_name=f"SCAR_{rec['lot_id']}.txt",
                        mime="text/plain",
                    )

    else:
        st.caption("왼쪽 사이드바의 '실시간 AI API 설정'에 키를 입력한 뒤, 아래 원문의 숫자나 단위(`nm`↔`um`, `wt%`↔`ppm`)를 수정해 실시간 검증할 수 있습니다.")
        c_in1, c_in2 = st.columns([1.3, 1])
        with c_in1:
            edited_coa = st.text_area("분석할 비정형 COA 원문", value=raw_text, height=280)
        with c_in2:
            live_sup = st.selectbox(
                "비교할 과거 이력 공급사",
                ["SUP-A", "SUP-B", "SUP-C", "SUP-D"],
                index=["SUP-A", "SUP-B", "SUP-C", "SUP-D"].index(rec["supplier_id"]),
            )
            live_lot_id = st.text_input("로트 번호 (Lot ID)", value=f"{rec['lot_id']}-LIVE")
            run_live = st.button("실시간 AI 에이전트 분석 실행 (Baseline vs v1 vs v2)", type="primary")

        if run_live:
            if not resolved_key:
                st.error("실시간 실행을 위해 왼쪽 사이드바 '실시간 AI API 설정'에 API 키를 입력하거나, '사전 검증 결과 즉시 재생' 모드를 이용하세요.")
            else:
                live_hist = get_supplier_history_before(live_sup, None, history_seed_df, gt_records)
                client = OpenAI(api_key=resolved_key, base_url=resolved_url) if resolved_url else OpenAI(api_key=resolved_key)
                with st.spinner(f"{resolved_model} 모델로 Baseline / Agent v1 / Agent v2 분석 중..."):
                    res_b = run_baseline_regex(edited_coa)
                    res_1 = run_agent_v1_simple_llm(edited_coa, client=client, model=resolved_model)
                    res_2 = run_agent_v2_self_correcting(
                        coa_text=edited_coa,
                        supplier_history_df=live_hist,
                        lot_id=live_lot_id,
                        supplier_name=live_sup,
                        client=client,
                        model=resolved_model,
                        generate_scar=True,
                    )

                st.markdown(
                    f"### 실시간 Agent v2 판정: {status_badge_html(res_2['predicted_status'])} "
                    f"(소요 {res_2['latency_sec']:.2f}초 · 자가수정 {res_2['retries_used']}회)",
                    unsafe_allow_html=True,
                )
                live_comp = pd.DataFrame(
                    [
                        {"방식": "Baseline (정규식)", "판정": res_b["predicted_status"], **res_b["extracted_values"]},
                        {"방식": "Agent v1 (단순 LLM)", "판정": res_1["predicted_status"], **res_1["extracted_values"]},
                        {"방식": "Agent v2 (자가수정·SPC)", "판정": res_2["predicted_status"], **res_2["extracted_values"]},
                    ]
                )
                st.dataframe(live_comp, hide_index=True)

                if res_2.get("eval_result"):
                    st.info(res_2["eval_result"]["decision_summary"])
                    fig_live = plot_single_lot_spc(
                        live_hist,
                        live_lot_id,
                        float(res_2["extracted_values"]["bet_m2g"]),
                        "bet_m2g",
                        live_sup,
                    )
                    st.pyplot(fig_live, clear_figure=True)

                with st.expander("실시간 Agent v2 실행 로그", expanded=True):
                    for line in res_2["trace"]:
                        st.markdown(f"- `{line}`")

                if res_2.get("scar_draft"):
                    with st.expander("자동 생성된 SCAR 초안", expanded=True):
                        st.text(res_2["scar_draft"])


# ===========================================================================
# PAGE 4: 검증 리포트 · AI가 틀린 것
# ===========================================================================
elif selected_page == "검증 리포트 · AI가 틀린 것":
    st.subheader("📊 40건 벤치마크 검증 리포트 및 '단순 AI(v1)가 틀린 것' 감사 기록")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "최종 판정 정확도 (Agent v2)",
        f"{row_v2['Overall_Decision_Accuracy_pct']:.1f}%",
        f"+{row_v2['Overall_Decision_Accuracy_pct'] - row_v1['Overall_Decision_Accuracy_pct']:.1f}%p vs v1",
    )
    m2.metric(
        "수치·단위 변환 정확도",
        f"{row_v2['Value_Unit_Accuracy_pct']:.1f}%",
        f"+{row_v2['Value_Unit_Accuracy_pct'] - row_v1['Value_Unit_Accuracy_pct']:.1f}%p vs v1",
    )
    m3.metric(
        "누적 SPC 드리프트 탐지율",
        f"{row_v2['SPC_Drift_Recall_pct']:.1f}%",
        f"+{row_v2['SPC_Drift_Recall_pct'] - row_v1['SPC_Drift_Recall_pct']:.1f}%p vs v1",
    )
    m4.metric(
        "건당 평균 처리 시간",
        f"{row_v2['Avg_Latency_sec']:.2f}초",
        f"{row_v2['Avg_Latency_sec'] - row_v1['Avg_Latency_sec']:+.2f}초 vs v1 (수기 15분 대비 99% 단축)",
    )

    st.markdown("#### 1) 시스템 아키텍처 개선 (v1 단순 LLM → v2 역할 분리·자가검증 루프)")
    a1, a2, a3, a4 = st.columns(4)
    a1.info("**Step 1. 원문 파싱 (LLM)**\n\n비정형 COA에서 측정값과 원문 단위를 암산 없이 그대로 JSON 추출")
    a2.info("**Step 2. 단위 환산·물리 검증 (Python)**\n\n`nm→μm`, `wt%→ppm` 강제 환산 후 물리 경계 이탈 시 최대 2회 자가 수정 요청")
    a3.info("**Step 3. 규격·누적 SPC 판정 (Python)**\n\n단건 규격 판정 + 공급사 과거 이력 DB 결합으로 $C_{pk}$ 및 연속 드리프트 탐지")
    a4.info("**Step 4. 계면 영향 해석·SCAR 발행**\n\n이상 로트의 슬러리·소결 공정 영향 도출 및 공급사 시정조치서 자동 작성")

    st.markdown("#### 2) 40건 정량 비교표 및 차트")
    st.dataframe(summary_df, hide_index=True)

    if os.path.exists("results/benchmark_comparison.png"):
        st.image("results/benchmark_comparison.png", caption="Baseline vs Agent v1 vs Agent v2 5대 핵심 지표 비교")

    st.divider()
    st.markdown(f"#### 3) 단순 AI(Agent v1)가 실제로 틀린 것 ({len(v1_audit_df)}건 감사 기록)")
    st.caption("단순 프롬프트(v1)의 오답을 정답지·원문과 대조해 원인을 분류하고, Agent v2에서 결정론적 Python 툴과 누적 DB로 전량 해결한 검증 기록입니다.")
    if not v1_audit_df.empty:
        st.dataframe(v1_audit_df, hide_index=True)

    st.divider()
    st.markdown("#### 4) 단건 검사로 놓치는 규격 내 SPC 드리프트 탐지 실증")
    g1, g2 = st.columns(2)
    with g1:
        if os.path.exists("results/spc_drift_sup_b_bet.png"):
            st.image("results/spc_drift_sup_b_bet.png", caption="SUP-B(일본) BET 비표면적 7연속 상향 드리프트 탐지")
    with g2:
        if os.path.exists("results/spc_drift_sup_d_d50.png"):
            st.image("results/spc_drift_sup_d_d50.png", caption="SUP-D(독일) D50 평균 입도 7연속 하향 드리프트 탐지")


# ===========================================================================
# PAGE 5: 자료실 (COA 원본 40건 + 과거 이력 DB + 정답지 열람/다운로드)
# ===========================================================================
elif selected_page == "자료실 (COA · DB)":
    st.subheader("📁 사내 품질 자료실 (원자료 조회 및 다운로드)")
    st.caption("벤치마크에 사용된 공급사 4곳의 비정형 COA 40건, 과거 승인 이력 DB, 정답지, 검증 결과표를 직접 확인하고 내려받을 수 있습니다.")

    dtab1, dtab2, dtab3 = st.tabs(["📄 분말 성적서 COA (40건)", "🗄️ 과거 입고 이력 DB (40로트)", "📋 정답지 · 벤치마크 결과표"])

    with dtab1:
        sup_filter = st.selectbox("공급사 필터", ["전체 (40건)", "SUP-A (한빛세라믹)", "SUP-B (Nippon Fine Powder)", "SUP-C (SinoMat)", "SUP-D (EuroOxide)"])
        filtered_gt = gt_records if sup_filter.startswith("전체") else [r for r in gt_records if r["supplier_id"] == sup_filter[:5]]

        pick_id = st.selectbox(
            "열람할 성적서 선택",
            [r["coa_id"] for r in filtered_gt],
            format_func=lambda cid: f"{cid} | {detailed_by_id[cid]['lot_id']} | {detailed_by_id[cid]['supplier_name']} | 판정: {detailed_by_id[cid]['true_status']}",
        )
        chosen = detailed_by_id[pick_id]
        coa_body = open(chosen["coa_path"], encoding="utf-8").read()
        st.code(coa_body, language="text")
        st.download_button(
            f"{pick_id}.txt 다운로드",
            data=coa_body.encode("utf-8"),
            file_name=f"{pick_id}.txt",
            mime="text/plain",
        )

    with dtab2:
        st.markdown("공급사별 과거 정상 승인 이력 시드 데이터 (`data/history_seed.csv`, 각 공급사 10로트 × 4곳 = 40건)")
        st.dataframe(history_seed_df, hide_index=True)
        st.download_button(
            "history_seed.csv 다운로드",
            data=history_seed_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="history_seed.csv",
            mime="text/csv",
        )

    with dtab3:
        gt_csv_df = pd.read_csv("data/ground_truth.csv")
        st.markdown("40건 COA 정답지 (`data/ground_truth.csv`)")
        st.dataframe(gt_csv_df, hide_index=True)
        c_dl1, c_dl2 = st.columns(2)
        c_dl1.download_button(
            "ground_truth.csv 다운로드",
            data=gt_csv_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="ground_truth.csv",
            mime="text/csv",
        )
        c_dl2.download_button(
            "benchmark_summary.csv 다운로드",
            data=summary_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="benchmark_summary.csv",
            mime="text/csv",
        )


# ===========================================================================
# PAGE 6: 교육 자료 · 규격서 (신입 IQC 엔지니어 매뉴얼)
# ===========================================================================
elif selected_page == "교육 자료 · 규격서":
    st.subheader("📘 신입 교육 자료 — 세라믹 원료 분말 IQC 및 계면·소결 공정 기초")
    st.caption("문서번호: CMX-QA-EDU-01 (개정 3판) · 발행: 품질보증팀 · 세라믹소재연구소")

    st.markdown("#### 1. 주력 원료 및 제조 공정 흐름")
    st.markdown(
        """
        당사가 제조하는 **연료전지용 고체산화물 전해질 기판(CMX-SOFC-E1)**과 정밀 세라믹 부품은 주원료 분말(**CMX-YSZ-05**)의 초기 물성이 최종 소결 밀도와 결함률을 좌우합니다.

        > **공정 흐름:** `원료 조달 (4개국)` → **`수입검사 · AI SPC 검증 (IQC)`** → `슬러리 분산·밀링` → `테이프 캐스팅 성형` → `건조·탈지 (400~600℃)` → `고온 소결 (1,450℃)` → `최종 출하 검사`
        """
    )

    st.markdown("#### 2. CMX-YSZ-05 4대 관리 물성 및 계면·소결 영향")
    spec_edu_df = pd.DataFrame(
        [
            {
                "관리 항목": "평균 입도 (D50)",
                "사내 규격": "0.40 ~ 0.60 μm",
                "물리 타당 범위": "0.05 ~ 50.0 μm",
                "공급사별 표기 변형": "nm, micron, d(0.5), Median Dia.",
                "규격 이탈·드리프트 시 후공정 결함": "조대화 시 비표면적·소결 구동력 저하(밀도 미달) / 미분화 시 응집 심화 및 소결 수축률 편차",
            },
            {
                "관리 항목": "비표면적 (BET)",
                "사내 규격": "5.50 ~ 7.50 m²/g",
                "물리 타당 범위": "0.50 ~ 100.0 m²/g",
                "공급사별 표기 변형": "SSA, Specific Surface Area, N2-BET",
                "규격 이탈·드리프트 시 후공정 결함": "표면 자유에너지 상승 -> 슬러리 점도 급상승, 분산 불량, 탈포 지연 및 소결체 내부 기공 결함",
            },
            {
                "관리 항목": "철 불순물 (Fe)",
                "사내 규격": "100.0 ppm 이하",
                "물리 타당 범위": "1.0 ~ 5000.0 ppm",
                "공급사별 표기 변형": "wt%, mg/kg, Iron impurity",
                "규격 이탈·드리프트 시 후공정 결함": "결정립계 편석으로 이상 입자 성장(abnormal grain growth) 및 고온 이온전도·절연 특성 저하",
            },
            {
                "관리 항목": "수분 함량 (Moisture)",
                "사내 규격": "0.50 wt% 이하",
                "물리 타당 범위": "0.01 ~ 5.0 wt%",
                "공급사별 표기 변형": "ppm, LOD (Loss on Drying), H2O",
                "규격 이탈·드리프트 시 후공정 결함": "분말 케이킹(caking) 및 성형체 건조·탈지 중 잔류 수분 급팽창에 의한 미세 균열(crack)",
            },
        ]
    )
    st.dataframe(spec_edu_df, hide_index=True)

    st.markdown("#### 3. IQC 판정 등급 및 SPC 경고 기준 (Western Electric 규칙 기반)")
    st.markdown(
        """
        *   **`PASS` (정상 입고):** 4개 항목 모두 사내 규격(USL/LSL) 이내이며, 공급사 누적 20로트 기준 공정능력($C_{pk} \\ge 1.00$) 및 추세 안정.
        *   **`SPC_WARNING` (조건부 입고 · 선제 SCAR):** 단건 규격은 통과했으나 **7로트 연속 중심선 편향(Run-7)**, **6로트 연속 단조 상승/하락(Trend-6)**, 또는 **$C_{pk} < 1.00$** 발생 시 선제 경고.
        *   **`FAIL` (입고 보류 · 반품/8D 요청):** 4개 항목 중 1개 이상 사내 규격 이탈(Spec Out). 즉시 격리 후 공급사 SCAR 발행.
        """
    )


# ===========================================================================
# PAGE 7: 회사 소개 · 사람 찾기
# ===========================================================================
elif selected_page == "회사 소개 · 사람 찾기":
    st.subheader("🏢 회사 소개 · 조직 및 사람 찾기")

    c_about1, c_about2 = st.columns([1.2, 1])
    with c_about1:
        st.markdown("#### Ceramix Materials㈜ (세라믹스 머티리얼즈 주식회사)")
        st.write(
            "2018년 설립된 정밀 산화물 세라믹 소재·부품 전문 기업입니다. "
            "연료전지 전해질 기판과 반도체·전장용 고신뢰성 소결 부품을 제조하며, "
            "ISO 9001:2015 및 IATF 16949 품질경영시스템을 기반으로 글로벌 4개국 원료 공급망을 운영합니다."
        )
        facts_df = pd.DataFrame(
            [
                ("주요 제품", "CMX-YSZ-05 고순도 분말 · CMX-SOFC-E1 전해질 기판 · CMX-AL99 정밀 소결 부품"),
                ("사업장", "본사·소재연구소(광주 북구 첨단과기로 208) / 제1공장(전남 화순군 화순읍 산단로 45)"),
                ("글로벌 원료 소싱", "한국 한빛세라믹(SUP-A) · 일본 Nippon Fine Powder(SUP-B) · 중국 SinoMat(SUP-C) · 독일 EuroOxide(SUP-D)"),
                ("품질 철학", "데이터로 원료를 검증하고, 규격 이탈 전 공정 드리프트를 선제 차단한다"),
            ],
            columns=["항목", "내용"],
        )
        st.dataframe(facts_df, hide_index=True)

    with c_about2:
        st.markdown("#### 핵심 조직 구성")
        org_df = pd.DataFrame(
            [
                ("품질보증팀", "원료 수입검사(IQC) · AI SPC 모니터링 · 공급사 SCAR 관리"),
                ("소결공정팀", "슬러리 분산 · 테이프 캐스팅 · 탈지 · 고온 정밀 소결"),
                ("세라믹소재연구소", "분말 물성·계면 거동 분석 · 신규 전해질 소재 개발"),
                ("구매자재팀", "해외 4개국 분말 조달 · 공급사 평가 및 이력 관리"),
                ("정보보안·DX팀", "세라믹스넷 사내 포털 운영 · AI 자동화 및 보안 감사"),
            ],
            columns=["부서", "주요 역할"],
        )
        st.dataframe(org_df, hide_index=True)

    st.divider()
    st.markdown("#### 👥 사람 찾기 (사내 연락처)")
    q = st.text_input("이름, 부서, 담당 업무로 검색", placeholder="예: 품질보증팀, 소결, SCAR")
    people_df = pd.DataFrame(PEOPLE)
    if q.strip():
        mask = people_df.apply(lambda row: q.strip().lower() in " ".join(row.astype(str)).lower(), axis=1)
        people_df = people_df[mask]
    st.dataframe(people_df, hide_index=True)


# ---------------------------------------------------------------------------
# 공통 하단 푸터
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="cm-footer">
      <strong>세라믹스넷 (CeramixNet) · Ceramix Materials Co., Ltd.</strong> &nbsp;|&nbsp;
      광주 첨단 R&D 본사 · 화순 제1공장 &nbsp;|&nbsp;
      정보보안·DX팀 내선 4100 · 품질보증팀 내선 4210 &nbsp;|&nbsp;
      GitHub 재현 코드: <code>dddhhh300/ceramix-coa-agent</code>
    </div>
    """,
    unsafe_allow_html=True,
)
