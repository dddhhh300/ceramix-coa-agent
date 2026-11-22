"""
app.py
Ceramix Materials - 세라믹 원료 분말 COA 검증 및 SPC 공정 이상 대응 AI 에이전트 웹사이트
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
    page_title="Ceramix Materials | AI IQC & SPC Agent",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .hero-box {
        background: linear-gradient(120deg, #0F172A 0%, #1E3A8A 65%, #0369A1 100%);
        color: #FFFFFF;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.02em;
    }
    .hero-sub {
        font-size: 1.0rem;
        color: #E2E8F0;
        margin: 0;
    }
    .badge-pass {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #DCFCE7; color: #166534; font-weight: 700; font-size: 0.9rem;
    }
    .badge-warn {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #FFEDD5; color: #9A3412; font-weight: 700; font-size: 0.9rem;
    }
    .badge-fail {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #FEE2E2; color: #991B1B; font-weight: 700; font-size: 0.9rem;
    }
    .badge-err {
        display: inline-block; padding: 4px 12px; border-radius: 999px;
        background-color: #E2E8F0; color: #334155; font-weight: 700; font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def status_badge_html(status: str) -> str:
    s = str(status).upper()
    if s == "PASS":
        return '<span class="badge-pass">PASS (정상 입고)</span>'
    if s == "SPC_WARNING":
        return '<span class="badge-warn">SPC_WARNING (공정 드리프트 경고)</span>'
    if s == "FAIL":
        return '<span class="badge-fail">FAIL (규격 이탈 · 입고 보류)</span>'
    return f'<span class="badge-err">{s}</span>'


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


history_seed_df, gt_records, summary_df, detailed_results = load_project_artifacts()
detailed_by_id = {r["coa_id"]: r for r in detailed_results}

with st.sidebar:
    st.markdown("### Ceramix Materials")
    st.caption("고순도 산화물 세라믹 분말·소결 부품 제조")
    st.divider()

    st.markdown("#### 실시간 API 설정 (선택)")
    st.caption("기본 '사전 실행 결과 즉시 재생' 모드는 API 키 없이 40건 전체 시연이 가능합니다.")
    sb_api_key = st.text_input("OpenAI 호환 API Key", type="password", placeholder="미입력 시 서버/로컬 환경변수 사용")
    sb_model = st.text_input("Model", value="", placeholder="기본: gpt-4o-mini")
    sb_base_url = st.text_input("Base URL (선택)", value="", placeholder="공식 OpenAI는 비워둠")

    resolved_key, resolved_url, resolved_model = resolve_api_config(sb_api_key, sb_base_url, sb_model)
    if resolved_key:
        st.success(f"실시간 API 사용 가능 ({resolved_model})")
    else:
        st.info("API 키 미설정: 사전 검증 결과 재생 모드로 즉시 시연 가능")

    st.divider()
    st.markdown("#### 핵심 규격 (CMX-YSZ-05)")
    st.markdown(
        """
        - **D50 (입도):** 0.40 ~ 0.60 μm
        - **BET (비표면적):** 5.50 ~ 7.50 m²/g
        - **Fe (불순물):** 100 ppm 이하
        - **Moisture (수분):** 0.50 wt% 이하
        """
    )

st.markdown(
    """
    <div class="hero-box">
      <div class="hero-title">Ceramix Materials Co., Ltd. — AI Quality Assurance Portal</div>
      <div class="hero-sub">
        세라믹 원료 분말(CMX-YSZ-05) 비정형 시험성적서(COA) 자동 검증 · 누적 SPC 드리프트 탐지 · 공급사 SCAR 대응 에이전트
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(
    [
        "1. 기업 소개 및 문제 정의",
        "2. AI 에이전트 라이브 데모",
        "3. 40건 벤치마크 검증 결과",
    ]
)

with tab1:
    col_a, col_b = st.columns([1.25, 1])

    with col_a:
        st.subheader("가상 기업 개요: Ceramix Materials (세라믹스 머티리얼즈)")
        st.write(
            "Ceramix Materials는 연료전지 전해질 기판 및 전자세라믹 부품을 생산하는 정밀 세라믹 제조 기업입니다. "
            "한국·일본·중국·독일 4개 원료 공급사로부터 핵심 산화물 분말(**CMX-YSZ-05**)을 조달해 "
            "슬러리 분산, 성형(테이프 캐스팅), 고온 소결 공정을 거쳐 완제품을 출하합니다."
        )

        st.markdown("#### 원료 분말 핵심 4대 물성과 계면·소결 공정 영향")
        spec_table = pd.DataFrame(
            [
                {
                    "검사 항목": "평균 입도 (D50)",
                    "표준 규격": "0.40 ~ 0.60 μm",
                    "공급사 표기 혼선 예시": "520 nm, d(0.5), Median Dia.",
                    "규격 이탈 시 공정 불량": "소결 구동력 저하(조대화) 또는 응집·수축률 편차(미분화)",
                },
                {
                    "검사 항목": "비표면적 (BET)",
                    "표준 규격": "5.50 ~ 7.50 m²/g",
                    "공급사 표기 혼선 예시": "SSA, Specific Surface Area, N2-BET",
                    "규격 이탈 시 공정 불량": "표면 자유에너지 상승 -> 슬러리 점도 급증·기공 결함",
                },
                {
                    "검사 항목": "철 불순물 (Fe)",
                    "표준 규격": "100 ppm 이하",
                    "공급사 표기 혼선 예시": "0.0085 wt%, mg/kg, Iron impurity",
                    "규격 이탈 시 공정 불량": "결정립계 편석으로 이상 입자 성장 및 전기적 특성 저하",
                },
                {
                    "검사 항목": "수분 함량 (Moisture)",
                    "표준 규격": "0.50 wt% 이하",
                    "공급사 표기 혼선 예시": "2400 ppm, LOD, H2O content",
                    "규격 이탈 시 공정 불량": "분말 케이킹 및 성형체 건조·탈지 중 미세 균열 발생",
                },
            ]
        )
        st.dataframe(spec_table, hide_index=True)

    with col_b:
        st.subheader("현장 병목 및 기존 접근법의 한계")
        st.error(
            "**현장 문제 (IQC 병목):** 4개국 공급사마다 COA 문서 양식·용어·단위가 제각각이라 "
            "수입검사 담당자가 수기로 단위 환산과 엑셀 입력을 수행하는 과정에서 오타·단위 실수·누적 공정 드리프트 누락이 발생합니다."
        )
        st.warning(
            "**기존 방식 1 — 고정 규칙/정규식 스크립트 (Baseline):** "
            "국내 표준 표 양식(SUP-A)만 인식하고 해외 공급사 3곳의 비정형 양식·용어 변화에서 전량 파싱 실패."
        )
        st.warning(
            "**기존 방식 2 — 단순 LLM 1회 호출 (Agent v1):** "
            "비정형 문서는 읽지만 암산 기반 단위 환산 오류와 규격 오판이 발생하며, "
            "단건 문서만 보므로 누적 SPC 드리프트를 전량 놓침."
        )
        st.success(
            "**제안 솔루션 — 자가검증·SPC 결합 AI 에이전트 (Agent v2):** "
            "역할 분리(LLM=비정형 원문 파싱 / Python=단위 환산·물리 범위 검증·누적 Cpk/SPC 계산)와 "
            "오류 시 최대 2회 자가 수정 루프로 정확도와 SPC 드리프트 탐지율을 대폭 개선."
        )

    st.divider()
    st.subheader("Agent v2 처리 아키텍처 (Tool-Augmented Self-Correcting Loop)")
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**Step 1. 비정형 파싱 (LLM)**\n\n공급사별 문서에서 4개 물성의 원문 숫자와 원문 단위를 암산 없이 그대로 JSON 추출")
    c2.info("**Step 2. 단위 표준화 & 물리 검증 (Python)**\n\n`nm→μm`, `wt%→ppm` 강제 환산 후 물리 경계 이탈 시 오류 힌트를 LLM에 반환해 최대 2회 재추출")
    c3.info("**Step 3. 규격 & 누적 SPC 분석 (Python)**\n\n단건 USL/LSL 대조 + 과거 승인 이력 DB 결합으로 공정능력지수($C_{pk}$) 및 7연속/6연속 드리프트 탐지")
    c4.info("**Step 4. 계면 영향 해석 & SCAR 발행**\n\n`FAIL` 또는 `SPC_WARNING` 시 후공정(슬러리·소결) 예상 영향 분석 및 공급사 시정조치서(SCAR) 자동 작성")

with tab2:
    st.subheader("비정형 COA 실시간 검증 및 SPC 이상 대응 데모")

    mode = st.radio(
        "실행 모드 선택",
        [
            "사전 검증 결과 즉시 재생 (40건 샘플 · API 키 불필요 · 채점 시연 권장)",
            "실시간 AI 에이전트 직접 실행 (텍스트 수정/신규 입력 · OpenAI 호환 API 호출)",
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
            st.caption(f"시나리오 태그: `{rec['scenario_tag']}` | 포함 함정: `{', '.join(rec['traps']) if rec['traps'] else '없음(표준)'}`")
            st.code(raw_text, language="text")

            st.markdown("#### 3가지 방식 판정 비교 (동일 문서)")
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
                        "방식": "Agent v2 (제안 에이전트)",
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
            st.markdown("#### Agent v2 분석 결과 및 추론 로그")
            st.markdown(
                f"최종 판정: {status_badge_html(v2['predicted_status'])} &nbsp;&nbsp; "
                f"| 처리 시간: **{v2['latency_sec']:.2f}초** | 자가 수정 횟수: **{v2['retries_used']}회**",
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

            with st.expander("단위 표준화 및 에이전트 단계별 실행 로그 (Trace) 보기", expanded=True):
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
        st.caption("아래 성적서 텍스트의 숫자나 단위(예: `nm` ↔ `um`, `wt%` ↔ `ppm`)를 자유롭게 수정한 뒤 실시간 분석을 실행해 보세요.")
        c_in1, c_in2 = st.columns([1.3, 1])
        with c_in1:
            edited_coa = st.text_area("분석할 비정형 COA 원문", value=raw_text, height=280)
        with c_in2:
            live_sup = st.selectbox(
                "비교할 과거 이력 공급사 컨텍스트",
                ["SUP-A", "SUP-B", "SUP-C", "SUP-D"],
                index=["SUP-A", "SUP-B", "SUP-C", "SUP-D"].index(rec["supplier_id"]),
            )
            live_lot_id = st.text_input("로트 번호 (Lot ID)", value=f"{rec['lot_id']}-LIVE")
            run_live = st.button("실시간 AI 에이전트 분석 실행 (Baseline vs v1 vs v2)", type="primary")

        if run_live:
            if not resolved_key:
                st.error("실시간 실행을 위해 왼쪽 사이드바에 OpenAI 호환 API 키를 입력하거나, 상단에서 '사전 검증 결과 즉시 재생' 모드를 선택하세요.")
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

with tab3:
    st.subheader("40건 비정형 COA 벤치마크 정량 검증 (요구사항 ④·⑤)")

    row_base = summary_df[summary_df["Method"] == "baseline"].iloc[0]
    row_v1 = summary_df[summary_df["Method"] == "agent_v1"].iloc[0]
    row_v2 = summary_df[summary_df["Method"] == "agent_v2"].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "최종 판정 정확도 (Agent v2)",
        f"{row_v2['Overall_Decision_Accuracy_pct']:.1f}%",
        f"+{row_v2['Overall_Decision_Accuracy_pct'] - row_v1['Overall_Decision_Accuracy_pct']:.1f}%p vs v1",
    )
    k2.metric(
        "수치·단위 변환 정확도",
        f"{row_v2['Value_Unit_Accuracy_pct']:.1f}%",
        f"+{row_v2['Value_Unit_Accuracy_pct'] - row_v1['Value_Unit_Accuracy_pct']:.1f}%p vs v1",
    )
    k3.metric(
        "누적 SPC 드리프트 탐지율",
        f"{row_v2['SPC_Drift_Recall_pct']:.1f}%",
        f"+{row_v2['SPC_Drift_Recall_pct'] - row_v1['SPC_Drift_Recall_pct']:.1f}%p vs v1",
    )
    k4.metric(
        "건당 평균 처리 시간",
        f"{row_v2['Avg_Latency_sec']:.2f}초",
        f"{row_v2['Avg_Latency_sec'] - row_v1['Avg_Latency_sec']:+.2f}초 vs v1 (수기 15분 대비 99% 단축)",
    )

    st.markdown("#### 1) 전체 성능 비교표 및 시각화")
    st.dataframe(summary_df, hide_index=True)

    if os.path.exists("results/benchmark_comparison.png"):
        st.image("results/benchmark_comparison.png", caption="Baseline vs Agent v1 vs Agent v2 5대 핵심 지표 비교")

    st.divider()
    st.markdown("#### 2) 단건 검사로 놓치는 규격 내 SPC 드리프트 탐지 실증")
    g1, g2 = st.columns(2)
    with g1:
        if os.path.exists("results/spc_drift_sup_b_bet.png"):
            st.image("results/spc_drift_sup_b_bet.png", caption="SUP-B(일본) BET 비표면적 7연속 상향 드리프트 탐지")
    with g2:
        if os.path.exists("results/spc_drift_sup_d_d50.png"):
            st.image("results/spc_drift_sup_d_d50.png", caption="SUP-D(독일) D50 평균 입도 7연속 하향 드리프트 탐지")

    st.divider()
    st.markdown("#### 3) 오류 분석 및 자가 수정 사례 탐색 (Case Inspector)")
    filter_opt = st.selectbox(
        "사례 필터",
        [
            "Agent v1은 틀렸지만 Agent v2가 해결한 건",
            "Agent v2 물리 검증 자가 수정(Retry) 발동 건",
            "Agent v2 잔여 오답 건 (추가 개선 과제)",
            "전체 40건 보기",
        ],
    )

    filtered_rows = []
    for r in detailed_results:
        v1_ok = r["agent_v1"]["status_exact"]
        v2_ok = r["agent_v2"]["status_exact"]
        retries = r["agent_v2"]["retries_used"]

        if filter_opt.startswith("Agent v1은") and not (not v1_ok and v2_ok):
            continue
        if filter_opt.startswith("Agent v2 물리") and retries == 0:
            continue
        if filter_opt.startswith("Agent v2 잔여") and v2_ok:
            continue

        filtered_rows.append(
            {
                "COA ID": r["coa_id"],
                "공급사": r["supplier_id"],
                "시나리오": r["scenario_tag"],
                "정답": r["true_status"],
                "Baseline": r["baseline"]["predicted_status"],
                "Agent v1": r["agent_v1"]["predicted_status"],
                "Agent v2": r["agent_v2"]["predicted_status"],
                "v2 자가수정": retries,
                "v1 수치정확": "O" if r["agent_v1"]["value_exact"] else "X",
                "v2 수치정확": "O" if r["agent_v2"]["value_exact"] else "X",
            }
        )

    if filtered_rows:
        st.dataframe(pd.DataFrame(filtered_rows), hide_index=True)
    else:
        st.info("선택한 조건에 해당하는 사례가 없습니다.")
