"""
PAX Identity EKG — Intent Classifier UI
─────────────────────────────────────────
Run with:  streamlit run app.py
"""

import os
import sys
import time
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
EXCEL_V3     = BASE_DIR / "data" / "intent_definitions_v3.xlsx"
VECTOR_STORE = BASE_DIR / "data" / "vector_store_v2"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PAX Identity EKG — Intent Classifier",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Confidence bar */
.conf-bar-wrap { background: #f0f0f0; border-radius: 6px; height: 10px; margin-top: 4px; }
.conf-bar      { height: 10px; border-radius: 6px; transition: width .4s; }

/* Entity pill */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 2px 3px;
}
.pill-role   { background:#dbeafe; color:#1e40af; }
.pill-tcode  { background:#dcfce7; color:#166534; }
.pill-auth   { background:#fef9c3; color:#854d0e; }
.pill-user   { background:#f3e8ff; color:#6b21a8; }
.pill-country{ background:#ffe4e6; color:#9f1239; }
.pill-dept   { background:#e0f2fe; color:#0c4a6e; }
.pill-sod    { background:#fce7f3; color:#831843; }
.pill-other  { background:#f1f5f9; color:#334155; }

/* Intent badge */
.intent-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: .3px;
}
.badge-green  { background:#dcfce7; color:#15803d; }
.badge-amber  { background:#fef9c3; color:#92400e; }
.badge-red    { background:#fee2e2; color:#991b1b; }
.badge-gray   { background:#f1f5f9; color:#475569; }

/* Similar example card */
.ex-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 13px;
}
.ex-score { color: #6366f1; font-weight: 600; }

/* Chat bubble */
.bubble-user {
    background: #eff6ff;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #1e3a5f;
}
.bubble-bot {
    background: #f0fdf4;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 14px;
    color: #14532d;
}

div[data-testid="stHorizontalBlock"] { align-items: flex-start; }
</style>
""", unsafe_allow_html=True)


# ── Cached resource loading ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading intent registry…")
def load_registry():
    from src.intent_registry import IntentRegistry
    return IntentRegistry(str(EXCEL_V3))

@st.cache_resource(show_spinner="Loading vector store…")
def load_store():
    from src.vector_store_v2 import IntentVectorStore
    return IntentVectorStore.load(str(VECTOR_STORE))

@st.cache_resource(show_spinner="Initialising classifier…")
def load_classifier(_registry, _store):
    from src.classifier_v2 import IntentClassifierV2
    return IntentClassifierV2(_registry, _store)


# ── Helpers ───────────────────────────────────────────────────────────────────

PERSONA_LABELS = {
    "END_USER":      "👤 End User (L4)",
    "PROCESS_OWNER": "🏢 Process Owner (L2)",
    "DATA_OWNER":    "🗄️ Data Owner (L2)",
    "APP_OWNER":     "💻 Application Owner (L3)",
    "AUDITOR":       "🔍 Auditor (L2/L3)",
}

INTENT_COLORS = {
    "VALIDATE_USER_REQUEST":          "badge-green",
    "DETECT_ROLES_REQUESTED":         "badge-green",
    "CHECK_FOR_SOD":                  "badge-amber",
    "ELABORATE_ROLE_FUNCTIONS":       "badge-green",
    "KNOWLEDGE_GRAPH_BASED_SCENARIOS":"badge-green",
    "RUN_A_WHATIF_SCENARIO":          "badge-amber",
    "OUT_OF_SCOPE":                   "badge-gray",
    "UNKNOWN":                        "badge-red",
}

ENTITY_PILL_CLASS = {
    "SAP_ROLE":    "pill-role",
    "TCODE":       "pill-tcode",
    "AUTH_OBJECT": "pill-auth",
    "USERNAME":    "pill-user",
    "COUNTRY":     "pill-country",
    "DEPARTMENT":  "pill-dept",
    "SOD_RULE_ID": "pill-sod",
}

def conf_color(c: float) -> str:
    if c >= 0.85: return "#22c55e"
    if c >= 0.70: return "#f59e0b"
    return "#ef4444"

def entity_pills_html(entities: dict) -> str:
    if not entities:
        return "<span style='color:#94a3b8;font-size:13px'>None extracted</span>"
    parts = []
    for key, val in entities.items():
        cls = ENTITY_PILL_CLASS.get(key, "pill-other")
        items = val if isinstance(val, list) else [val]
        for item in items:
            parts.append(f'<span class="pill {cls}"><b>{key}</b>: {item}</span>')
    return " ".join(parts)

def setup_check() -> bool:
    return (VECTOR_STORE / "index.faiss").exists()


# ── Setup screen ──────────────────────────────────────────────────────────────

def show_setup_screen():
    st.error("⚠️ Vector store not found. Run setup first.", icon="🚧")
    st.code("python main_v2.py setup", language="bash")
    st.info("This embeds all sample queries from the intent sheet into a FAISS index. Takes ~1 minute on first run.")
    if st.button("🔄 Run setup now", type="primary"):
        with st.spinner("Building vector store from intent_definitions_v3.xlsx …"):
            try:
                from src.intent_registry import IntentRegistry
                from src.vector_store_v2 import build_vector_store
                registry = IntentRegistry(str(EXCEL_V3))
                build_vector_store(registry, save_path=str(VECTOR_STORE))
                st.success("✅ Setup complete! Reloading…")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Setup failed: {e}")


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(registry):
    with st.sidebar:
        st.image("https://img.shields.io/badge/PAX-Identity%20EKG-6366f1?style=for-the-badge", width=200)
        st.markdown("## ⚙️ User Context")

        persona = st.selectbox(
            "Persona",
            options=list(PERSONA_LABELS.keys()),
            format_func=lambda x: PERSONA_LABELS[x],
            key="persona",
        )

        username = st.text_input("Username", value="pjunker", key="username")
        department = st.selectbox(
            "Department",
            ["Procurement & Supply Chain", "Finance", "Manufacturing",
             "Sales & Marketing", "Human Resources", "IT", "Audit & Compliance"],
            key="department",
        )
        country = st.selectbox(
            "Country",
            ["United Kingdom", "Germany", "United States",
             "France", "India", "Australia", "Netherlands"],
            key="country",
        )
        level = st.select_slider("User Level", options=["L4","L3","L2","L1"], value="L4", key="level")

        st.divider()
        st.markdown("## 📋 Intent Registry")
        for record in registry.records:
            with st.expander(f"**{record.intent_name}** `{record.coverage_pct or ''}`"):
                st.markdown(f"**ID:** `{record.intent_id}`")
                st.markdown(f"**Personas:** {', '.join(record.allowed_personas)}")
                st.markdown("**Sub-intents:**")
                for si in record.sub_intents:
                    st.markdown(f"- {si.index}) {si.label}")
                st.markdown(f"**Entity types:** `{', '.join(record.entity_types)}`")

        st.divider()
        if st.button("🗑️ Clear chat history"):
            st.session_state.history = []
            st.rerun()

        return persona, username, department, country, level


# ── Result card ───────────────────────────────────────────────────────────────

def render_result_card(result):
    badge_cls = INTENT_COLORS.get(result.intent_id, "badge-gray")
    gate_icon = "✅" if result.persona_gate_passed else "🚫"
    clarify_icon = "🤔" if result.requires_clarification else "✅"

    # Top row: intent badge + confidence
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f'<span class="intent-badge {badge_cls}">{result.intent_id}</span>',
            unsafe_allow_html=True,
        )
        if result.intent_name:
            st.caption(result.intent_name)
    with col2:
        c = result.confidence
        st.markdown(f"**Confidence**")
        st.markdown(
            f'<div class="conf-bar-wrap"><div class="conf-bar" style="width:{c*100:.0f}%;background:{conf_color(c)}"></div></div>'
            f'<div style="font-size:13px;font-weight:600;color:{conf_color(c)}">{c*100:.0f}%</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Three columns: sub-intent / gates / entities
    c1, c2, c3 = st.columns([2, 1, 2])

    with c1:
        st.markdown("**🎯 Sub-Intent**")
        st.info(result.sub_intent or "—")
        st.markdown("**💬 Reasoning**")
        st.caption(result.reasoning or "—")

    with c2:
        st.markdown("**🔐 Gates**")
        st.markdown(f"{gate_icon} Persona gate")
        st.markdown(f"{clarify_icon} Clarification")
        st.markdown(f"**⏱️ Time**")
        st.markdown(f"`{result.processing_time_ms:.0f} ms`")

    with c3:
        st.markdown("**🏷️ Extracted Entities**")
        st.markdown(entity_pills_html(result.extracted_entities), unsafe_allow_html=True)

    # Clarification question
    if result.requires_clarification and result.clarification_question:
        st.warning(f"🤔 **Bot would ask:** {result.clarification_question}")

    # Bot response template
    if result.bot_response_template:
        with st.expander("📄 Bot response template (Col F match)"):
            st.markdown(result.bot_response_template)

    # Retrieved examples
    if result.retrieval_info:
        with st.expander(f"🔍 Top {len(result.retrieval_info)} similar examples retrieved"):
            for ex in result.retrieval_info:
                score_pct = int(ex["score"] * 100)
                st.markdown(
                    f'<div class="ex-card">'
                    f'<span class="ex-score">{score_pct}% match</span> &nbsp;·&nbsp; '
                    f'<code>{ex["intent_id"]}</code><br>'
                    f'<span style="color:#475569">{ex["text"]}</span>'
                    f'<br><span style="color:#94a3b8;font-size:11px">sub: {ex.get("sub_intent","")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    if not setup_check():
        show_setup_screen()
        return

    # Load resources
    registry   = load_registry()
    store      = load_store()
    classifier = load_classifier(registry, store)

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []

    # Sidebar
    persona, username, department, country, level = render_sidebar(registry)

    # Main area
    st.markdown("# 🔐 PAX Identity EKG — Intent Classifier")
    st.caption("Type a natural language query as a Teams user. The classifier detects intent, sub-intent, and extracts SAP entities.")

    # Tab layout
    tab_chat, tab_batch, tab_debug = st.tabs(["💬 Chat", "📦 Batch Test", "🛠️ Debug"])

    # ── Chat tab ──────────────────────────────────────────────────────────────
    with tab_chat:
        # Render history
        for item in st.session_state.history:
            st.markdown(
                f'<div class="bubble-user">👤 <b>{item["persona"]}</b>: {item["query"]}</div>',
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                render_result_card(item["result"])
            st.markdown("")

        # Input
        with st.form("query_form", clear_on_submit=True):
            col_inp, col_btn = st.columns([5, 1])
            with col_inp:
                query = st.text_input(
                    "Your message",
                    placeholder="e.g. Please give me access to ZC:P2P:PO_CREATOR________:1000",
                    label_visibility="collapsed",
                )
            with col_btn:
                submitted = st.form_submit_button("Send ➤", type="primary", use_container_width=True)

        # Quick examples
        st.markdown("**Quick examples:**")
        ex_cols = st.columns(3)
        examples = [
            "I have joined the procurement team in the UK",
            "Please give me access to ZC:P2P:PO_CREATOR________:1000",
            "Do I have any SoD violations?",
            "What does the PO-Creator role allow me to do?",
            "If I remove ME28 from this role will the conflict disappear?",
            "Show all access risks in the SAP application",
        ]
        for i, ex in enumerate(examples):
            with ex_cols[i % 3]:
                if st.button(ex, key=f"ex_{i}", use_container_width=True):
                    query = ex
                    submitted = True

        if submitted and query.strip():
            with st.spinner("🔄 Classifying…"):
                result = classifier.classify(
                    message=query.strip(),
                    authenticated_user=username,
                    persona=persona,
                    user_level=level,
                    department=department,
                    country=country,
                )

            st.session_state.history.append({
                "query": query.strip(),
                "persona": PERSONA_LABELS[persona],
                "result": result,
            })
            st.rerun()

    # ── Batch test tab ────────────────────────────────────────────────────────
    with tab_batch:
        st.markdown("### Run all test cases")
        st.caption("Tests all 6 intents + persona gate + out-of-scope handling.")

        test_cases = [
            ("I have just joined the procurement team in the UK",                        "END_USER",      "VALIDATE_USER_REQUEST"),
            ("I am leaving the organisation please revoke all my SAP accesses",          "END_USER",      "VALIDATE_USER_REQUEST"),
            ("I want to find out what roles my peers have",                              "END_USER",      "DETECT_ROLES_REQUESTED"),
            ("Please give me access to this role: ZC:P2P:PO_CREATOR________:1000",      "END_USER",      "DETECT_ROLES_REQUESTED"),
            ("I am a purchaser and need permission to create Purchase orders",           "END_USER",      "DETECT_ROLES_REQUESTED"),
            ("I would like to understand why this role assignment is flagged as a risk", "AUDITOR",       "CHECK_FOR_SOD"),
            ("Can you explain what the functions of each of these roles are",            "END_USER",      "ELABORATE_ROLE_FUNCTIONS"),
            ("Please can you provide me with a summary of all accesses my direct reports have", "END_USER","KNOWLEDGE_GRAPH_BASED_SCENARIOS"),
            ("I want to assess all access risks in the SAP application",                 "AUDITOR",       "KNOWLEDGE_GRAPH_BASED_SCENARIOS"),
            ("If I remove ME29 and release auth objects from ZC:P2P:PO_CREATOR________:1000 will the SoD conflict disappear?", "PROCESS_OWNER", "RUN_A_WHATIF_SCENARIO"),
            ("If I remove ME28 from this role will the conflict go away?",               "END_USER",      "OUT_OF_SCOPE"),   # persona gate
            ("What is the weather today?",                                               "END_USER",      "OUT_OF_SCOPE"),
        ]

        if st.button("▶️ Run all tests", type="primary"):
            results_data = []
            progress = st.progress(0, text="Running…")
            for i, (msg, test_persona, expected) in enumerate(test_cases):
                r = classifier.classify(
                    message=msg,
                    authenticated_user=username,
                    persona=test_persona,
                    user_level="L4",
                    department=department,
                    country=country,
                )
                passed = r.intent_id == expected
                results_data.append({
                    "✓": "✅" if passed else "❌",
                    "Query": msg[:60] + ("…" if len(msg) > 60 else ""),
                    "Persona": test_persona,
                    "Expected": expected,
                    "Got": r.intent_id,
                    "Conf": f"{r.confidence*100:.0f}%",
                    "Time": f"{r.processing_time_ms:.0f}ms",
                })
                progress.progress((i + 1) / len(test_cases), text=f"{i+1}/{len(test_cases)} done")

            progress.empty()
            passed_count = sum(1 for r in results_data if r["✓"] == "✅")
            st.success(f"**{passed_count}/{len(test_cases)} passed**")
            st.dataframe(results_data, use_container_width=True, hide_index=True)

    # ── Debug tab ─────────────────────────────────────────────────────────────
    with tab_debug:
        st.markdown("### Raw classifier output")
        debug_query = st.text_area(
            "Query to debug",
            value="Please give me access to ZC:P2P:PO_CREATOR________:1000",
            height=80,
        )
        debug_persona = st.selectbox(
            "Persona", list(PERSONA_LABELS.keys()),
            format_func=lambda x: PERSONA_LABELS[x],
            key="debug_persona",
        )
        if st.button("🔬 Classify & show raw JSON"):
            with st.spinner("Classifying…"):
                r = classifier.classify(
                    message=debug_query,
                    authenticated_user=username,
                    persona=debug_persona,
                    user_level=level,
                    department=department,
                    country=country,
                )
            st.json(r.to_dict())
            st.markdown("**Retrieved examples:**")
            st.json(r.retrieval_info)


if __name__ == "__main__":
    main()