import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
import os
import json
import re
from datetime import datetime
import pdfplumber
import docx
from io import BytesIO

# ------------------ CLIENTS ------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Resources Network - Look for Expert",
    page_icon="🔍",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("DP_BG1.png", width=150)
st.write("")
with col2:
    st.markdown(
        "<h2 style='text-align: center;'>🌐 Resources Network - Look for Expert</h2>",
        unsafe_allow_html=True
    )

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")
ai_model = st.sidebar.radio(
    "Choose AI Model for Recommendations:",
    options=["GPT-4o Mini (OpenAI)", "Claude (Anthropic)"],
    index=1
)
st.sidebar.markdown("---")
if ai_model == "GPT-4o Mini (OpenAI)":
    st.sidebar.info("Using **OpenAI GPT-4o Mini** for recommendations.")
else:
    st.sidebar.info("Using **Anthropic Claude** for recommendations.")

# ------------------ FILE UPLOADERS ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Upload Business Document")
st.sidebar.caption("Upload your pitch deck or business plan to improve expert matching.")
founder_uploaded_file = st.sidebar.file_uploader(
    "Pitch Deck / Business Document",
    type=["pdf", "docx", "txt"],
    key="founder_doc"
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Check an Expert's Profile")
st.sidebar.caption("Upload an expert's resume to score them against your requirement.")
expert_uploaded_file = st.sidebar.file_uploader(
    "Expert Resume / Profile",
    type=["pdf", "docx", "txt"],
    key="expert_profile"
)

# ------------------ CLEAR CHAT BUTTON ------------------
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat & History"):
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.last_query = ""
    st.session_state.pending_retry = False
    st.session_state.retry_query = ""
    st.session_state.search_history = []
    st.rerun()

# ------------------ SIDEBAR: RECENT SEARCHES ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🕘 Recent Searches")

if not st.session_state.get("search_history"):
    st.sidebar.caption("No searches yet. Your recent prompts will appear here.")
else:
    recent_prompts = list(reversed(st.session_state.search_history))[:10]
    for i, item in enumerate(recent_prompts):
        label = item["query"][:40] + ("…" if len(item["query"]) > 40 else "")
        meta  = f"🏆{item['tier1']} · 🔍{item['tier2']}  {item['timestamp']}"
        if st.sidebar.button(
            f"↩ {label}",
            key=f"sidebar_rerun_{i}",
            use_container_width=True,
            help=item["query"]   # full query on hover
        ):
            st.session_state._rerun_query = item["query"]
            st.rerun()
        st.sidebar.caption(meta)

# ------------------ DOCUMENT EXTRACTION UTILS ------------------
def extract_text_from_pdf_bytes(file_bytes):
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        text = ""
    return text.strip()

def extract_text_from_docx_bytes(file_bytes):
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception:
        text = ""
    return text.strip()

def extract_text_from_uploaded_file(uploaded_file):
    if uploaded_file is None:
        return ""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf_bytes(file_bytes)
    elif uploaded_file.type in [
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword"
    ]:
        return extract_text_from_docx_bytes(file_bytes)
    elif uploaded_file.type == "text/plain":
        return file_bytes.decode("utf-8", errors="ignore")
    return ""

def extract_text_from_file_path(file_path):
    if not file_path or not isinstance(file_path, str) or file_path.strip() == "":
        return ""
    file_path = file_path.strip()
    try:
        if file_path.lower().endswith(".pdf"):
            with open(file_path, "rb") as f:
                return extract_text_from_pdf_bytes(f.read())
        elif file_path.lower().endswith(".docx"):
            with open(file_path, "rb") as f:
                return extract_text_from_docx_bytes(f.read())
        elif file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        return ""
    return ""

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel(
        "mentors.xlsx",
        engine="openpyxl",
        dtype=str,
        na_filter=False
    )

    required_cols = [
        "Expertise", "Secondary Expertise", "Industry", "Secondary Industry",
        "Description", "Expertise Tags", "Industry Tags",
        "Document Path", "LinkedIn", "Qualification",
        "Current Organization", "Current Designation",
        "Program", "Years of Experience",
        # ── New problem-statement columns ──
        "What is the one business problem you are most qualified to advise on from direct experience?",
        "Other Experience(s), if any",
        "Industry - Operator Data",
        "What revenue stage do you understand best from the inside? (Select one only)",
        "Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") == "streamlit" or \
               os.environ.get("IS_STREAMLIT_CLOUD") is not None or \
               not os.path.exists(os.path.expanduser("~/.streamlit/config.toml"))

    if is_cloud:
        df["Doc Text"] = ""
    else:
        df["Doc Text"] = df["Document Path"].apply(extract_text_from_file_path)

    # Parse "Industry - Operator Data" — pipe-separated, positional
    # Position maps to the 15 industry labels in order
    _INDUSTRY_LABELS = [
        "Agri / Food Processing", "Manufacturing", "Healthcare", "Climate Tech",
        "Deep Tech", "Enterprise Tech", "D2C / B2C", "Services",
        "Fintech & Financial Services", "Automotive & Auto Components",
        "Media & Entertainment", "HR Services", "Legal Services",
        "Transportation & Logistics", "Other (please specify)",
    ]
    def parse_operator_industries(val):
        parts = [p.strip() for p in str(val).split("|")]
        active = [
            _INDUSTRY_LABELS[i]
            for i, p in enumerate(parts)
            if p and p.lower() not in ("", "nan", "0", "no", "false") and i < len(_INDUSTRY_LABELS)
        ]
        return ", ".join(active) if active else ""

    df["Active Industries"] = df["Industry - Operator Data"].apply(parse_operator_industries)

    # Helper: safe column read (returns empty string if column missing)
    def col(name):
        return df[name] if name in df.columns else ""

    df["combined"] = (
        "Expertise: " + col("Expertise") + ". " +
        "Secondary Expertise: " + col("Secondary Expertise") + ". " +
        "Industry: " + col("Industry") + ". " +
        "Secondary Industry: " + col("Secondary Industry") + ". " +
        "Active Industry Sectors: " + col("Active Industries") + ". " +
        "Description: " + col("Description") + ". " +
        "Tags: " + col("Expertise Tags") + " " + col("Industry Tags") + ". " +
        "Qualification: " + col("Qualification") + ". " +
        "Current Organization: " + col("Current Organization") + ". " +
        "Current Designation: " + col("Current Designation") + ". " +
        "Core Business Problem Advised: " + col("What is the one business problem you are most qualified to advise on from direct experience?") + ". " +
        "Other Experiences: " + col("Other Experience(s), if any") + ". " +
        "Revenue Stage Expertise: " + col("What revenue stage do you understand best from the inside? (Select one only)") + ". " +
        "Growth Ceiling Story: " + col("Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)")
    )
    # Ensure every combined value is a clean plain string — no NaN, no None, no floats
    df["combined"] = df["combined"].fillna("").astype(str).str.strip()
    return df

df = load_data()

# ── Lookup dicts built once at startup — fast O(1) access per card ──
program_lookup    = df.set_index("Name")["Program"].to_dict()           if "Program"            in df.columns else {}
experience_lookup = df.set_index("Name")["Years of Experience"].to_dict() if "Years of Experience" in df.columns else {}
linkedin_lookup   = df.set_index("Name")["LinkedIn"].to_dict()           if "LinkedIn"           in df.columns else {}

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ CREATE VECTORS ------------------
@st.cache_data
def get_vectors(texts):
    # Guarantee every item is a non-empty plain string before encoding
    clean = [str(t).strip() if t and str(t).strip() else "no information available" for t in texts]
    return model.encode(clean, batch_size=64, show_progress_bar=False)

vectors = get_vectors(df["combined"].tolist())

# ------------------ PROGRAM FILTER ------------------
# Placed here so df and vectors are both available.
# Sidebar widgets defined after df is loaded — Streamlit renders them top-to-bottom
# in the sidebar regardless of where in the script they are written.

_all_programs = sorted(set(
    p.strip()
    for val in df["Program"]
    for p in str(val).split(",")
    if p.strip() and p.strip().lower() not in ("nan", "none", "")
))

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Filter by Program")
st.sidebar.caption("Leave empty to search all mentors.")

selected_programs = st.sidebar.multiselect(
    "Program(s)",
    options=_all_programs,
    default=[],
    key="program_filter",
    placeholder="All programs"
)

if selected_programs:
    _prog_mask = df["Program"].apply(
        lambda x: any(
            sel.strip() in [p.strip() for p in str(x).split(",")]
            for sel in selected_programs
        )
    )
    filtered_df = df[_prog_mask].reset_index(drop=True)
    filtered_vectors = get_vectors(filtered_df["combined"].tolist())
    st.sidebar.info(
        f"🔎 **{len(filtered_df)}** mentor(s) in: "
        f"{', '.join(selected_programs)}"
    )
else:
    filtered_df = df
    filtered_vectors = vectors
    st.sidebar.caption(f"🔎 Searching all **{len(df)}** mentors")

# ------------------ EXTRACT UPLOADED FILE TEXTS ------------------
founder_doc_text = ""
if founder_uploaded_file:
    founder_doc_text = extract_text_from_uploaded_file(founder_uploaded_file)
    if founder_doc_text:
        st.sidebar.success("✅ Business document parsed!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from business document.")

expert_profile_text = ""
if expert_uploaded_file:
    expert_profile_text = extract_text_from_uploaded_file(expert_uploaded_file)
    if expert_profile_text:
        st.sidebar.success("✅ Expert profile parsed!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from expert profile.")

# ------------------ SESSION STATE INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "pending_retry" not in st.session_state:
    st.session_state.pending_retry = False
if "retry_query" not in st.session_state:
    st.session_state.retry_query = ""
if "search_history" not in st.session_state:
    st.session_state.search_history = []  # max 10 items, {query, timestamp, tier1, tier2}

# ------------------ SAVE TO HISTORY HELPER ------------------
def save_to_history(query, tier1_count, tier2_count):
    """Saves a search to history. Keeps only the last 10 unique queries."""
    # Remove duplicate if same query already exists
    st.session_state.search_history = [
        h for h in st.session_state.search_history
        if h["query"].strip().lower() != query.strip().lower()
    ]
    st.session_state.search_history.append({
        "query": query,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "tier1": tier1_count,
        "tier2": tier2_count,
    })
    # Keep only latest 10
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[-10:]

# ------------------ INTENT DETECTION ------------------
def detect_intent(user_input, last_recommendations):
    followup_keywords = [
        "tell me more", "more about", "compare", "vs", "versus",
        "which is better", "difference between", "what about",
        "explain", "elaborate", "details about", "why is", "how is",
        "refine", "show more", "different", "another", "instead",
        "same industry", "similar", "like #", "expert #", "first expert",
        "second expert", "third expert", "top expert", "number"
    ]
    input_lower = user_input.lower()
    has_recommendations = len(last_recommendations) > 0
    is_followup = has_recommendations and any(
        kw in input_lower for kw in followup_keywords
    )
    return "followup" if is_followup else "new_search"

# ------------------ AI CALL HELPER ------------------
def call_ai(prompt, max_tokens=2048):
    if ai_model == "GPT-4o Mini (OpenAI)":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    else:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            temperature=0,
            system="""You are an AI expert-matching assistant for Resources Network,
helping Indian founders find the most suitable experts from a curated database.

CORE RULES YOU ALWAYS FOLLOW:
- Industry match alone is NOT enough for Tier 1
- Hands-on operator experience alone is NOT enough for Tier 1
- BOTH must be present for Tier 1
- When in doubt → Tier 2, not Tier 1
- Never assume industry match if not clearly stated in the profile
- Be honest — do not mark Yes for hands-on just because the expert is impressive

MINIMUM RESULT RULES — CRITICAL:
- You MUST always return at least 1 result total across both tiers
- If no Tier 1 experts exist, return the best available expert(s) as Tier 2
- If no strong matches exist at all, return the closest 1-2 experts as Tier 2
  with an honest Match Reason explaining the partial fit
- Never return an empty array — always surface the best available option
- Always respond in the language and format specified in the user message""",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# ------------------ PROGRAM BADGE HELPER ------------------
def render_program_badge(expert_name):
    prog_val = program_lookup.get(expert_name, "").strip()
    if not prog_val or prog_val.lower() in ("", "nan", "none"):
        return
    prog_list = [p.strip() for p in prog_val.split(",") if p.strip()]
    if not prog_list:
        return
    badges_html = " ".join([
        f"<span style='background:#1F4E79;color:white;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;margin-right:4px;font-weight:500;'>"
        f"📌 {p}</span>"
        for p in prog_list
    ])
    st.markdown(
        f"<div style='margin-bottom:8px;'>"
        f"<span style='font-size:13px;color:#555;font-weight:500;'>Program: </span>"
        f"{badges_html}</div>",
        unsafe_allow_html=True
    )

# ------------------ EXPERIENCE BADGE HELPER ------------------
def render_experience_badge(expert_name):
    exp_val = experience_lookup.get(expert_name, "").strip()
    if not exp_val or exp_val.lower() in ("", "nan", "none"):
        return
    st.markdown(
        f"<div style='margin-bottom:12px;'>"
        f"<span style='font-size:13px;color:#555;font-weight:500;'>Experience: </span>"
        f"<span style='background:#E2EFDA;color:#375623;padding:3px 10px;"
        f"border-radius:12px;font-size:12px;font-weight:500;'>"
        f"🏅 {exp_val}</span></div>",
        unsafe_allow_html=True
    )

# ------------------ DISPLAY SINGLE EXPERT CARD ------------------
def display_expert_card(expert, index, tier_label, source_df):
    hands_on = expert.get("Hands On Experience", "").strip()
    if hands_on == "Yes":
        badge = "🟢 Hands-On/Operator"
    elif hands_on == "Partial":
        badge = "🟡 Partial Hands-on/Operator Experience"
    else:
        badge = "🔴 No Direct Experience"

    overall   = expert.get("Overall Score", "N/A")
    expert_name = expert.get("Name", "N/A")

    with st.expander(
        f"#{index} — {expert_name} | ⭐ {overall}/10 | {badge}",
        expanded=(index == 1)
    ):
        # ── Meta badges row ──
        render_program_badge(expert_name)
        render_experience_badge(expert_name)

        st.markdown("### 📊 Match Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)

        for col, key, label, max_val in [
            (sc1, "Industry Match Score", "🏭 Industry Match", "3"),
            (sc2, "Hands On Score", "🛠️ Hands-on/Operator Experience", "3"),
            (sc3, "Expertise Score", "💼 Expertise", "2"),
            (sc4, "Credibility Score", "🏅 Key Credentials", "2"),
        ]:
            with col:
                raw = expert.get(key, "N/A")
                parts = (
                    raw.split("|")
                    if isinstance(raw, str) and "|" in raw
                    else [raw, ""]
                )
                st.metric(label, f"{parts[0].strip()} / {max_val}")
                if len(parts) > 1:
                    st.caption(parts[1].strip())

        st.markdown("---")

        st.markdown("**🎯 Core Area of Expertise**")
        st.write(expert.get("Core Expertise", "Not available"))

        st.markdown("**✅ Why Suitable**")
        st.write(expert.get("Match Reason", ""))

        st.markdown("**💼 Relevant Experience**")
        st.write(expert.get("Relevant Experience", ""))

        st.markdown("**🛠️ Hands-on/Operator Experience in Founder's Area**")
        if hands_on == "Yes":
            st.success(f"✅ Yes — Hands-on/Operator — {expert.get('Hands On Details', '')}")
        elif hands_on == "Partial":
            st.warning(f"⚠️ Partial Hands-on/Operator — {expert.get('Hands On Details', '')}")
        else:
            st.error(f"❌ No Hands-on/Operator Experience — {expert.get('Hands On Details', '')}")

        # ── New insight fields (only shown if present) ──
        core_problem_match = expert.get("Core Problem Match", "").strip()
        revenue_fit        = expert.get("Revenue Stage Fit", "").strip()
        growth_relevance   = expert.get("Growth Ceiling Relevance", "").strip()

        show_new = any(
            v and v.lower() not in ("not specified", "n/a", "")
            for v in [core_problem_match, revenue_fit, growth_relevance]
        )

        if show_new:
            st.markdown("---")
            st.markdown("**🎯 Problem & Stage Fit**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if core_problem_match and core_problem_match.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**🔑 Core Problem Match**")
                    st.caption(core_problem_match)
            with col_b:
                if revenue_fit and revenue_fit.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**📈 Revenue Stage Fit**")
                    st.caption(revenue_fit)
            with col_c:
                if growth_relevance and growth_relevance.lower() not in ("not specified", "n/a", ""):
                    st.markdown("**🚀 Growth Ceiling Relevance**")
                    st.caption(growth_relevance)

        linkedin = linkedin_lookup.get(expert_name, "")
        if linkedin and str(linkedin).strip() != "":
            st.markdown(f"[🔗 View LinkedIn Profile]({linkedin})")

# ------------------ DISPLAY FULL RESULTS ------------------
def display_expert_results(ai_recommendations, source_df):
    if not ai_recommendations:
        return

    tier1 = [m for m in ai_recommendations if m.get("Tier") == "1"]
    tier2 = [m for m in ai_recommendations if m.get("Tier") == "2"]

    if tier1:
        st.markdown("""
        ## 🏆 Tier 1 — Strong Matches
        > These experts match **both the industry AND have hands-on operator experience**
        in the founder's problem area.
        """)
        for i, expert in enumerate(tier1):
            display_expert_card(expert, i + 1, "Tier 1", source_df)
    else:
        st.warning(
            "⚠️ No Tier 1 matches found — no expert matched both industry "
            "AND operator experience for this requirement."
        )

    if tier2:
        st.markdown("---")
        st.markdown("""
        ## 🔍 Tier 2 — Partial Matches
        > These experts match **either the industry OR have relevant experience**
        but not both. They may still provide useful guidance.
        """)
        for i, expert in enumerate(tier2):
            display_expert_card(expert, i + 1, "Tier 2", source_df)

# ------------------ CORE SEARCH FUNCTION ------------------
# ── Stopwords for keyword extraction ──────────────────────────────────
_SEARCH_STOPWORDS = {
    "looking", "experts", "expert", "need", "help", "want", "mentor",
    "mentors", "industry", "with", "from", "that", "have", "for", "the",
    "and", "who", "can", "are", "best", "good", "find", "show", "give",
    "someone", "person", "people", "startup", "business", "company",
    "founder", "their", "this", "about", "know", "does", "has"
}

def extract_keywords(query):
    """Extract meaningful keywords from the query for exact-match boosting."""
    words = re.findall(r'\b[\w&/]+\b', query.lower())
    return [w for w in words if len(w) > 2 and w not in _SEARCH_STOPWORDS]

def run_search(query, source_df, source_vectors):
    enriched_query = query
    if founder_doc_text:
        enriched_query = (
            f"{query}\n\n"
            f"Context from business document:\n{founder_doc_text[:1500]}"
        )

    # ── Step 1: Semantic similarity ──────────────────────────────────────
    query_vec = model.encode([enriched_query])
    similarity = cosine_similarity(query_vec, source_vectors)
    source_df = source_df.copy()
    source_df["semantic_score"] = similarity[0]

    # ── Step 2: Keyword boost ─────────────────────────────────────────────
    # For each keyword in the query, award +0.08 per exact match found
    # anywhere in the mentor's combined text. This rescues niche terms
    # (e.g. "HORECA", "D2C", "SaaS") that have weak semantic embeddings.
    keywords = extract_keywords(query)
    if keywords:
        def keyword_score(combined_text):
            text_lower = str(combined_text).lower()
            return sum(0.08 for kw in keywords if kw in text_lower)
        source_df["keyword_boost"] = source_df["combined"].apply(keyword_score)
    else:
        source_df["keyword_boost"] = 0.0

    # ── Step 3: Combined score — semantic + keyword boost ─────────────────
    source_df["score"] = source_df["semantic_score"] + source_df["keyword_boost"]

    # ── Step 4: Sort and take top 30 (expanded from 20 for better recall) ─
    candidates = source_df.sort_values(
        by=["score", "Name"],
        ascending=[False, True]
    ).head(30)

    expert_info = ""
    for _, row in candidates.iterrows():
        doc_summary = (
            row["Doc Text"][:500]
            if row["Doc Text"] and len(row["Doc Text"]) > 50
            else "Not available"
        )
        # Collect active industry sectors
        active_sectors = row.get("Active Industries", "").strip()

        # New problem-statement fields (show only if non-empty)
        core_problem  = row.get("What is the one business problem you are most qualified to advise on from direct experience?", "").strip()
        other_exp     = row.get("Other Experience(s), if any", "").strip()
        revenue_stage = row.get("What revenue stage do you understand best from the inside? (Select one only)", "").strip()
        growth_story  = row.get("Describe one time you helped a business break through a growth ceiling.* (What was the ceiling, and what specifically changed?)", "").strip()

        def r(col_name):
            """Safe row getter — returns empty string if column missing."""
            v = row.get(col_name, "")
            return str(v).strip() if v and str(v).strip().lower() not in ("nan", "0") else ""

        expert_info += f"""
Name: {row['Name']}
Expertise: {r('Expertise')}
Secondary Expertise: {r('Secondary Expertise')}
Industry: {r('Industry')}
Active Industry Sectors: {active_sectors if active_sectors else 'Not specified'}
Current Designation: {r('Current Designation')}
Current Organization: {r('Current Organization')}
Qualification: {r('Qualification')}
Description: {r('Description')}
Core Business Problem They Can Advise On: {core_problem if core_problem else 'Not specified'}
Other Experiences: {other_exp if other_exp and other_exp != '0' else 'Not specified'}
Revenue Stage Expertise: {revenue_stage if revenue_stage else 'Not specified'}
Growth Ceiling Story: {growth_story[:400] if growth_story else 'Not specified'}
Document Summary: {doc_summary}
---
"""

    founder_context_section = ""
    if founder_doc_text:
        founder_context_section = f"""
Founder also uploaded a business document:
{founder_doc_text[:1500]}
"""

    previous_context = ""
    if st.session_state.last_query and st.session_state.last_query != query:
        previous_context = f"""
Note: The founder previously searched for: "{st.session_state.last_query}"
This is a new search. Treat it independently but keep previous context in mind.
"""

    prompt = f"""
You are helping an Indian founder find the right expert.

Founder's business brief and problem statement:
"{query}"

{founder_context_section}
{previous_context}

Here are expert profiles to evaluate:
{expert_info}

TIER CLASSIFICATION RULES — This is the most important part:

NEW DATA AVAILABLE — USE THESE FIELDS FOR STRONGER MATCHING:
- "Core Business Problem They Can Advise On": This is the single problem the expert
  is MOST qualified to advise on from direct experience. If this aligns with the
  founder's problem → strong signal for Tier 1.
- "Active Industry Sectors": Checkbox-based sectors the expert has worked in.
  Use this alongside "Industry" for industry matching.
- "Revenue Stage Expertise": The revenue stage the expert understands best from
  the inside. Match this to where the founder's business is.
- "Growth Ceiling Story": A real example of how the expert helped a business
  break through a growth ceiling. If the ceiling described matches the founder's
  problem → very strong Tier 1 signal.
- "Other Experiences": Additional experiences beyond core expertise.

TIER 1 — Strong Match (show minimum 1, maximum 5):
An expert qualifies for Tier 1 ONLY if BOTH conditions are true:
  ✅ Condition 1 — Industry Match: The expert has directly worked IN the same
     or very closely related industry as the founder's business. Use both the
     Industry field AND Active Industry Sectors for this assessment.
  ✅ Condition 2 — Operator Experience: The expert has PERSONALLY done the
     specific task or solved the specific problem the founder is facing.
     Use "Core Business Problem They Can Advise On" and "Growth Ceiling Story"
     as primary evidence for this condition.

If even ONE condition is missing → expert goes to Tier 2, NOT Tier 1.
Be strict. It is better to show 1 Tier 1 expert than to incorrectly
promote a weak match to Tier 1.

TIER 2 — Partial Match (maximum 5):
Experts who meet at least ONE of the following:
  - Matches the industry but lacks operator experience in the specific problem
  - Has operator experience in the problem area but from a different industry
  - Has strong relevant expertise that could still be useful to the founder

MINIMUM RESULT GUARANTEE — MANDATORY:
- You MUST return at least 1 expert total across both tiers
- If no expert qualifies for Tier 1, place the best available expert(s) in Tier 2
- If no strong matches exist, return the closest 1–2 experts as Tier 2 with an
  honest Match Reason that clearly states the fit is partial or indirect
- NEVER return an empty array under any circumstances

SCORING (apply to all experts regardless of tier):
- Industry Match: 3 points
- Operator Experience: 3 points
- Relevant Expertise: 2 points
- Key Credentials: 2 points

STRICT RULES:
- Never put an expert in Tier 1 just because they are impressive or well-qualified
- Industry match alone is NOT enough for Tier 1
- Operator experience alone is NOT enough for Tier 1
- BOTH must be present for Tier 1
- Be honest about Operator Experience: Yes = personally done it,
  Partial = advised/consulted on it, No = no relevant experience
- Be CONSISTENT — if an expert's profile does not clearly state they worked
  in this industry, do NOT assume it
- If you are not sure whether an expert qualifies for Tier 1, put them in Tier 2
- When in doubt → Tier 2, not Tier 1

Return strictly as a JSON array. Include ALL experts evaluated —
Tier 1 first (1-5 experts), then Tier 2 (up to 5 experts).
Total array must have at least 1 object and at most 10 objects.

Format:
[
  {{
    "Tier": "1",
    "Name": "expert name exactly as given",
    "Overall Score": "score out of 10 as number only e.g. 8",
    "Industry Match Score": "score | one line explanation e.g. 3 | Worked in manufacturing exports for 10 years",
    "Hands On Score": "score | one line explanation e.g. 3 | Personally handled DGFT and LC documentation",
    "Expertise Score": "score | one line explanation e.g. 2 | Strong supply chain expertise",
    "Credibility Score": "score | one line explanation — focus on industry recognition, awards, board memberships, publications or speaking engagements e.g. 2 | Featured speaker at CII, board member at 2 startups",
    "Core Expertise": "1 line — the single most relevant core area of expertise this expert is known for",
    "Match Reason": "2-3 lines — lead with WHY they qualify for this tier based on industry + operator experience",
    "Relevant Experience": "specific experience directly relevant to the founder's problem",
    "Current Designation": "their current designation",
    "Current Organization": "their current organization",
    "Qualification": "their qualification",
    "Hands On Experience": "Yes / No / Partial",
    "Hands On Details": "1-2 lines on what they have personally done as an operator. If No/Partial, state clearly what is missing.",
    "Core Problem Match": "1 line — how their stated core business problem aligns with the founder's need (or 'Not specified' if field is empty)",
    "Revenue Stage Fit": "1 line — the revenue stage this expert understands best and whether it matches the founder's stage (or 'Not specified' if field is empty)",
    "Growth Ceiling Relevance": "1 line — whether their growth ceiling story is relevant to the founder's challenge (or 'Not specified' if field is empty)"
  }}
]

Return only the JSON array. No extra text, no markdown outside the array.
"""

    ai_raw = call_ai(prompt, max_tokens=3000)
    cleaned = re.sub(r"```json|```", "", ai_raw).strip()

    try:
        ai_recommendations = json.loads(cleaned)
    except json.JSONDecodeError:
        match_json = re.search(r'\[.*\]', cleaned, re.DOTALL)
        ai_recommendations = (
            json.loads(match_json.group()) if match_json else []
        )

    # ── PYTHON-LEVEL FALLBACK ──
    if not ai_recommendations:
        for _, row in candidates.head(2).iterrows():
            ai_recommendations.append({
                "Tier": "2",
                "Name": row.get("Name", "Unknown"),
                "Overall Score": "4",
                "Industry Match Score": "1 | Closest available match — no strong industry alignment found",
                "Hands On Score": "1 | Limited operator experience confirmed for this requirement",
                "Expertise Score": "1 | Some relevant expertise may apply",
                "Credibility Score": "1 | Profile available for review",
                "Core Expertise": row.get("Expertise", "Not specified"),
                "Match Reason": (
                    "No strong match found for this requirement. "
                    "This expert is the closest available in the network based on semantic similarity. "
                    "Suitability should be verified manually before outreach."
                ),
                "Relevant Experience": row.get("Description", "")[:300],
                "Current Designation": row.get("Current Designation", ""),
                "Current Organization": row.get("Current Organization", ""),
                "Qualification": row.get("Qualification", ""),
                "Hands On Experience": "No",
                "Hands On Details": (
                    "Hands-on fit for this specific requirement could not be confirmed. "
                    "Please review the full profile before proceeding."
                )
            })

    return ai_recommendations

# ------------------ TABS ------------------
tab_search, tab_session = st.tabs([
    "🔍 Expert Search",
    "🧠 Session Intelligence"
])

# ── Helper: render a full session analysis ─────────────────────────────────
def _render_session_analysis(a, snap, mentor_name):
    health = a.get("overall_session_health", {})
    health_score = health.get("score", "N/A")
    health_rec = health.get("recommendation", "")

    rec_colors = {
        "Continue": "success", "Reconnect": "success",
        "Needs Follow-up": "warning", "Try Different Mentor": "error"
    }
    getattr(st, rec_colors.get(health_rec, "info"))(
        f"**🏥 Session Health: {health_score}/10** | Recommendation: **{health_rec}**\n\n"
        f"{health.get('summary', '')}"
    )
    st.markdown("---")

    with st.expander("1️⃣ What Was the Ask?", expanded=True):
        ask = a.get("original_ask", {})
        st.markdown(f"**{ask.get('summary', 'Not available')}**")
        conf = ask.get("confidence", "")
        conf_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf, "⚪")
        st.caption(f"Confidence: {conf_color} {conf}")

    with st.expander("2️⃣ Mentor's Expertise", expanded=True):
        exp = a.get("mentor_expertise", {})
        st.write(exp.get("summary", "Not available"))
        fit_val = exp.get("relevant_to_ask", "")
        getattr(st, {"Yes": "success", "Partial": "warning", "No": "error"}.get(fit_val, "info"))(
            f"Relevant to Ask: **{fit_val}**"
        )
        demonstrated = exp.get("expertise_demonstrated", [])
        if demonstrated:
            st.markdown("**Expertise Demonstrated:**")
            for item in (demonstrated if isinstance(demonstrated, list) else [demonstrated]):
                st.markdown(f"• {item}")

    with st.expander("3️⃣ Discussion & Action Items", expanded=True):
        disc = a.get("session_discussion", {})
        acts = a.get("actionable_items", {})
        st.markdown("**Discussion Summary**")
        st.write(disc.get("summary", "Not available"))
        topics = disc.get("key_topics", [])
        if topics:
            st.markdown("**Key Topics:**")
            cols_t = st.columns(min(len(topics), 3))
            for i, topic in enumerate(topics):
                with cols_t[i % 3]:
                    st.markdown(
                        f"<span style='background:#EBF5FB;color:#1A5276;padding:4px 10px;"
                        f"border-radius:8px;font-size:13px;display:inline-block;margin:2px;'>"
                        f"📌 {topic}</span>", unsafe_allow_html=True
                    )
        st.markdown("---")
        st.markdown("**✅ Action Items for Venture**")
        for item in acts.get("for_venture", []):
            st.markdown(f"☑️ {item}")
        mentor_acts = [m for m in acts.get("for_mentor", []) if str(m).strip()]
        if mentor_acts:
            st.markdown("**🔁 Mentor Follow-up**")
            for item in mentor_acts:
                st.markdown(f"☑️ {item}")
        clarity = acts.get("clarity_score", "")
        timeline = acts.get("timeline_mentioned", "Not mentioned")
        clarity_color = {"Clear": "🟢", "Vague": "🟡", "None": "🔴"}.get(clarity, "⚪")
        st.caption(f"Timeline: {timeline} | Clarity: {clarity_color} {clarity}")

    with st.expander("4️⃣ Engagement Signals", expanded=True):
        eng = a.get("engagement_signals", {})
        interest = eng.get("interest_level", "")
        getattr(st, {"High": "success", "Medium": "warning", "Low": "error"}.get(interest, "info"))(
            f"Interest Level: **{interest}**"
        )
        reconnect = eng.get("wants_to_reconnect", "Unclear")
        reconnect_ev = eng.get("reconnect_evidence", "Not mentioned")
        st.markdown(f"**Wants to Reconnect:** {reconnect}")
        if reconnect_ev and reconnect_ev != "Not mentioned":
            st.info(f"💬 *\"{reconnect_ev}\"*")
        for sig in eng.get("signals", []):
            st.markdown(f"• **{sig.get('signal', '')}**")
            ref = sig.get("transcript_reference", "")
            if ref and ref != "Not available from transcript":
                st.markdown(
                    f"<blockquote style='border-left:3px solid #3498DB;padding:6px 12px;"
                    f"background:#EBF5FB;border-radius:4px;font-size:13px;color:#1A5276;'>"
                    f"📝 <em>{ref}</em></blockquote>", unsafe_allow_html=True
                )

    with st.expander("5️⃣ Service Request Signals", expanded=True):
        srv = a.get("service_request_signals", {})
        if srv.get("detected") == "Yes":
            st.warning(f"⚠️ **Service Request Detected** — {srv.get('type', 'Unknown')}")
            st.write(srv.get("details", ""))
            for sig in srv.get("signals", []):
                st.markdown(f"• **{sig.get('signal', '')}**")
                ref = sig.get("transcript_reference", "")
                if ref and ref != "Not available from transcript":
                    st.markdown(
                        f"<blockquote style='border-left:3px solid #E67E22;padding:6px 12px;"
                        f"background:#FEF9E7;border-radius:4px;font-size:13px;color:#784212;'>"
                        f"📝 <em>{ref}</em></blockquote>", unsafe_allow_html=True
                    )
        else:
            st.success("✅ No service/paid engagement requests detected.")

    with st.expander("6️⃣ Venture Feedback", expanded=True):
        fb = a.get("venture_feedback", {})
        sentiment = fb.get("overall_sentiment", "")
        getattr(st, {
            "Very Positive": "success", "Positive": "success",
            "Neutral": "info", "Mixed": "warning", "Negative": "error"
        }.get(sentiment, "info"))(f"Overall Sentiment: **{sentiment}**")
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            st.metric("Rating Given", fb.get("rating_mentioned", "Not mentioned"))
        with fb_col2:
            st.metric("Follow-up Requested", fb.get("followup_requested", "Unclear"))
        st.markdown("**Fit Assessment**")
        st.write(fb.get("fit_assessment", "Not available"))
        praise = fb.get("specific_praise", "")
        if praise and praise != "Not available":
            st.success(f"👍 {praise}")
        concerns = fb.get("concerns_or_gaps", "")
        if concerns and concerns != "None mentioned":
            st.warning(f"⚠️ {concerns}")

    st.markdown("---")
    # Export
    export_lines = [
        "SESSION INTELLIGENCE REPORT", "=" * 50,
        f"Mentor: {snap.get('mentor', 'N/A')} | Venture: {snap.get('venture', 'N/A')}",
        f"Date: {snap.get('date') or 'Not specified'} | Type: {snap.get('type', 'N/A')}",
        "", f"OVERALL HEALTH: {health_score}/10 | Recommendation: {health_rec}",
        health.get("summary", ""), "",
        "1. ORIGINAL ASK", a.get("original_ask", {}).get("summary", "N/A"), "",
        "2. MENTOR EXPERTISE", a.get("mentor_expertise", {}).get("summary", "N/A"),
        f"Relevant to Ask: {a.get('mentor_expertise', {}).get('relevant_to_ask', 'N/A')}", "",
        "3. SESSION DISCUSSION", a.get("session_discussion", {}).get("summary", "N/A"),
        "", "ACTION ITEMS FOR VENTURE:",
    ]
    for item in a.get("actionable_items", {}).get("for_venture", []):
        export_lines.append(f"  - {item}")
    export_lines += [
        "", "4. ENGAGEMENT SIGNALS",
        f"Interest Level: {a.get('engagement_signals', {}).get('interest_level', 'N/A')}",
        f"Wants to Reconnect: {a.get('engagement_signals', {}).get('wants_to_reconnect', 'N/A')}",
        "", "5. SERVICE REQUEST SIGNALS",
        f"Detected: {a.get('service_request_signals', {}).get('detected', 'No')}",
        a.get("service_request_signals", {}).get("details", ""), "",
        "6. VENTURE FEEDBACK",
        f"Sentiment: {a.get('venture_feedback', {}).get('overall_sentiment', 'N/A')}",
        f"Fit Assessment: {a.get('venture_feedback', {}).get('fit_assessment', 'N/A')}",
    ]
    mentor_slug = snap.get("mentor", "mentor").replace(" ", "_")
    date_slug = (snap.get("date") or "nodateXX").replace(" ", "_")
    st.download_button(
        label="📥 Download Report as TXT",
        data="\n".join(export_lines),
        file_name=f"session_report_{mentor_slug}_{date_slug}.txt",
        mime="text/plain",
        use_container_width=True
    )



# ==================== TAB 2: SESSION INTELLIGENCE ====================
with tab_session:
    st.markdown("## 🧠 Session Intelligence — Mentor Session Analyser")
    st.caption(
        "Track and analyse all sessions for a mentor. Each mentor can have multiple sessions "
        "across different ventures. All sessions are stored and viewable in the history panel."
    )

    # ── Session state ──
    if "si_sessions" not in st.session_state:
        # Dict keyed by mentor name → list of session records
        # Each record: {session_id, venture, date, type, notes, feedback,
        #               transcript_text, analysis, created_at}
        st.session_state.si_sessions = {}
    if "si_active_session_id" not in st.session_state:
        st.session_state.si_active_session_id = None
    if "si_view_mode" not in st.session_state:
        # "new_session" | "view_session"
        st.session_state.si_view_mode = "new_session"

    # ── Build mentor list from df ──
    mentor_list = sorted([
        n for n in df["Name"].dropna().unique()
        if str(n).strip() and str(n).strip().lower() not in ("nan", "none", "")
    ])

    # ── Build venture list: try ventures.xlsx, else derive from session history ──
    @st.cache_data
    def load_ventures():
        import os
        if os.path.exists("ventures.xlsx"):
            try:
                vdf = pd.read_excel("ventures.xlsx", dtype=str, na_filter=False)
                for col in ["Venture Name", "Name", "Company", "Startup"]:
                    if col in vdf.columns:
                        names = sorted([
                            v for v in vdf[col].dropna().unique()
                            if str(v).strip() and str(v).strip().lower() not in ("nan", "none", "")
                        ])
                        if names:
                            return names
            except Exception:
                pass
        return []

    static_venture_list = load_ventures()

    def get_venture_list():
        """Combine static list with any ventures added via sessions."""
        from_sessions = [
            s["venture"]
            for sessions in st.session_state.si_sessions.values()
            for s in sessions
            if s.get("venture", "").strip()
        ]
        combined = sorted(set(static_venture_list + from_sessions))
        return combined

    # ═══════════════════════════════════════════════════════════
    # LAYOUT: Left sidebar panel (mentor + session history) | Main content
    # ═══════════════════════════════════════════════════════════
    si_left, si_right = st.columns([1, 2.4])

    with si_left:
        st.markdown("### 👤 Select Mentor")
        selected_mentor = st.selectbox(
            "Mentor",
            options=["— Select a mentor —"] + mentor_list,
            key="si_selected_mentor",
            label_visibility="collapsed"
        )

        if selected_mentor and selected_mentor != "— Select a mentor —":
            mentor_sessions = st.session_state.si_sessions.get(selected_mentor, [])

            st.markdown("---")
            st.markdown(f"#### 📂 Sessions for {selected_mentor.split()[0]}")

            # ── New Session button ──
            if st.button("➕ Add New Session", use_container_width=True, key="si_new_btn"):
                st.session_state.si_view_mode = "new_session"
                st.session_state.si_active_session_id = None
                st.rerun()

            st.markdown("")

            if not mentor_sessions:
                st.caption("No sessions recorded yet.")
            else:
                # Show sessions newest first
                for sess in reversed(mentor_sessions):
                    sid = sess["session_id"]
                    label_venture = sess.get("venture", "Unknown Venture")
                    label_date = sess.get("date", "")
                    label_date_str = f" · {label_date}" if label_date else ""
                    sess_num = mentor_sessions.index(sess) + 1
                    health_score = ""
                    if sess.get("analysis"):
                        score = sess["analysis"].get("overall_session_health", {}).get("score", "")
                        if score:
                            health_score = f" ⭐{score}"

                    is_active = st.session_state.si_active_session_id == sid
                    btn_label = (
                        f"{'▶ ' if is_active else ''}Session {sess_num}{health_score}\n"
                        f"**{label_venture}**{label_date_str}"
                    )
                    if st.button(
                        f"{'▶ ' if is_active else ''}Session {sess_num}{health_score} — {label_venture}{label_date_str}",
                        key=f"si_sess_btn_{sid}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        st.session_state.si_active_session_id = sid
                        st.session_state.si_view_mode = "view_session"
                        st.rerun()

                    # Delete button inline
                    if st.button(f"🗑️ Delete", key=f"si_del_{sid}", use_container_width=True):
                        st.session_state.si_sessions[selected_mentor] = [
                            s for s in mentor_sessions if s["session_id"] != sid
                        ]
                        if st.session_state.si_active_session_id == sid:
                            st.session_state.si_active_session_id = None
                            st.session_state.si_view_mode = "new_session"
                        st.rerun()

                    st.markdown("")

    # ── Main content panel ──
    with si_right:

        if not selected_mentor or selected_mentor == "— Select a mentor —":
            st.info("👈 Select a mentor from the panel on the left to get started.")

        elif st.session_state.si_view_mode == "view_session" and st.session_state.si_active_session_id:
            # ───────────────────────────────────────────────────
            # VIEW EXISTING SESSION ANALYSIS
            # ───────────────────────────────────────────────────
            mentor_sessions = st.session_state.si_sessions.get(selected_mentor, [])
            active_sess = next(
                (s for s in mentor_sessions if s["session_id"] == st.session_state.si_active_session_id),
                None
            )

            if not active_sess:
                st.warning("Session not found. Please select another session.")
            else:
                a = active_sess.get("analysis")
                snap = {
                    "mentor": selected_mentor,
                    "venture": active_sess.get("venture", ""),
                    "date": active_sess.get("date", ""),
                    "type": active_sess.get("type", ""),
                }
                sess_idx = mentor_sessions.index(active_sess) + 1

                st.markdown(
                    f"### 📋 Session {sess_idx} — {snap['venture'] or 'Unknown Venture'}"
                    + (f" · {snap['date']}" if snap['date'] else "")
                )
                st.caption(f"Mentor: {selected_mentor} | Type: {snap['type']} | Recorded: {active_sess.get('created_at', '')}")

                if not a:
                    st.warning("This session has no AI analysis yet. Notes were saved but analysis was not run.")
                    if active_sess.get("notes"):
                        st.markdown("**Notes:**")
                        st.write(active_sess["notes"])
                else:
                    # ── Render full analysis (same render function defined below) ──
                    _render_session_analysis(a, snap, selected_mentor)

        else:
            # ───────────────────────────────────────────────────
            # NEW SESSION FORM
            # ───────────────────────────────────────────────────
            mentor_sessions = st.session_state.si_sessions.get(selected_mentor, [])
            next_sess_num = len(mentor_sessions) + 1

            st.markdown(f"### ➕ New Session #{next_sess_num} for {selected_mentor}")

            # ── Venture selector ──
            venture_options = get_venture_list()
            si_col1, si_col2 = st.columns(2)

            with si_col1:
                if venture_options:
                    venture_choice = st.selectbox(
                        "🏢 Select Venture *",
                        options=["— Select a venture —"] + venture_options + ["+ Add new venture..."],
                        key="si_venture_select"
                    )
                    if venture_choice == "+ Add new venture...":
                        si_venture_name = st.text_input(
                            "Enter venture name",
                            key="si_venture_new",
                            placeholder="e.g. GreenCrop Agritech"
                        )
                    elif venture_choice == "— Select a venture —":
                        si_venture_name = ""
                    else:
                        si_venture_name = venture_choice
                else:
                    si_venture_name = st.text_input(
                        "🏢 Venture Name *",
                        key="si_venture_text",
                        placeholder="e.g. GreenCrop Agritech"
                    )

            with si_col2:
                si_session_date = st.text_input(
                    "📅 Session Date",
                    key="si_session_date",
                    placeholder="e.g. 15 May 2025 (optional)"
                )

            si_session_type = st.selectbox(
                "📌 Session Type",
                ["1×1 Expert Connect", "Group Masterclass", "Intro Call", "Follow-up Session", "Other"],
                key="si_session_type"
            )

            si_fc1, si_fc2 = st.columns(2)

            with si_fc1:
                st.markdown("**📋 Meeting Notes / Summary**")
                si_meeting_notes = st.text_area(
                    "Meeting notes",
                    height=160,
                    key="si_meeting_notes",
                    label_visibility="collapsed",
                    placeholder="Paste any notes from the session — agenda, discussion points, decisions..."
                )

            with si_fc2:
                st.markdown("**💬 Venture Feedback**")
                si_feedback = st.text_area(
                    "Venture feedback",
                    height=160,
                    key="si_feedback",
                    label_visibility="collapsed",
                    placeholder="Paste the venture's feedback about this mentor session..."
                )

            st.markdown("**📄 Upload Session Transcript / Documents**")
            up_col1, up_col2 = st.columns(2)
            with up_col1:
                si_transcript_file = st.file_uploader(
                    "Session Transcript (PDF, DOCX, TXT)",
                    type=["pdf", "docx", "txt"],
                    key="si_transcript"
                )
            with up_col2:
                si_additional_file = st.file_uploader(
                    "Additional Document (optional)",
                    type=["pdf", "docx", "txt"],
                    key="si_additional_doc"
                )

            # Parse files
            si_transcript_text = ""
            if si_transcript_file:
                si_transcript_text = extract_text_from_uploaded_file(si_transcript_file)
                if si_transcript_text:
                    st.success(f"✅ Transcript parsed — {len(si_transcript_text.split())} words")
                else:
                    st.warning("⚠️ Could not extract text from transcript.")

            si_additional_text = ""
            if si_additional_file:
                si_additional_text = extract_text_from_uploaded_file(si_additional_file)

            # ── Action buttons ──
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                save_only = st.button(
                    "💾 Save Notes Only",
                    use_container_width=True,
                    key="si_save_btn"
                )
            with btn_col2:
                analyze_clicked = st.button(
                    "🚀 Save & Analyse",
                    type="primary",
                    use_container_width=True,
                    key="si_analyze_btn"
                )

            # ── Handle save / analyse ──
            if save_only or analyze_clicked:
                if not si_venture_name or si_venture_name.strip() == "":
                    st.error("Please select or enter a venture name.")
                elif not si_meeting_notes and not si_feedback and not si_transcript_text:
                    st.error("Please provide at least meeting notes, feedback, or a transcript.")
                else:
                    import uuid as _uuid
                    new_sid = str(_uuid.uuid4())[:8]
                    new_record = {
                        "session_id": new_sid,
                        "venture": si_venture_name.strip(),
                        "date": si_session_date.strip(),
                        "type": si_session_type,
                        "notes": si_meeting_notes,
                        "feedback": si_feedback,
                        "transcript_text": si_transcript_text,
                        "analysis": None,
                        "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    }

                    if analyze_clicked:
                        with st.spinner("Analysing session with AI..."):
                            context_parts = [
                                f"MENTOR NAME: {selected_mentor}",
                                f"VENTURE NAME: {si_venture_name}",
                            ]
                            if si_session_date:
                                context_parts.append(f"SESSION DATE: {si_session_date}")
                            context_parts.append(f"SESSION TYPE: {si_session_type}")
                            if si_meeting_notes:
                                context_parts.append(f"\nMEETING NOTES / SUMMARY:\n{si_meeting_notes}")
                            if si_feedback:
                                context_parts.append(f"\nVENTURE FEEDBACK:\n{si_feedback}")
                            if si_transcript_text:
                                context_parts.append(f"\nSESSION TRANSCRIPT:\n{si_transcript_text[:12000]}")
                            if si_additional_text:
                                context_parts.append(f"\nADDITIONAL DOCUMENT:\n{si_additional_text[:3000]}")

                            full_context = "\n\n".join(context_parts)

                            si_prompt = f"""You are an expert program analyst for a startup accelerator's Resources Network team.
You have been given information about a mentor-venture session. Analyse everything carefully and return a structured JSON response.

SESSION DATA:
{full_context}

Your task is to extract and analyse the following 7 dimensions. For each section, if a transcript is provided, include DIRECT QUOTES from the transcript as evidence.

Return ONLY a valid JSON object with exactly these keys:

{{
  "original_ask": {{
    "summary": "What was the specific ask or problem for which the mentor was connected to this venture? 2-3 sentences.",
    "confidence": "High / Medium / Low"
  }},
  "mentor_expertise": {{
    "summary": "What is the mentor's expertise area as understood from this session? 2-3 sentences.",
    "relevant_to_ask": "Yes / Partial / No",
    "expertise_demonstrated": ["area 1", "area 2", "area 3"]
  }},
  "session_discussion": {{
    "summary": "What was actually discussed? 3-5 sentences covering main topics.",
    "key_topics": ["topic 1", "topic 2", "topic 3"],
    "actionable_outcome": "Yes / No"
  }},
  "actionable_items": {{
    "for_venture": ["action 1", "action 2"],
    "for_mentor": ["mentor follow-up 1"],
    "timeline_mentioned": "Any deadline mentioned, or 'Not mentioned'",
    "clarity_score": "Clear / Vague / None"
  }},
  "engagement_signals": {{
    "interest_level": "High / Medium / Low",
    "signals": [
      {{
        "signal": "description of engagement signal",
        "transcript_reference": "quote from transcript if available, else 'Not available from transcript'"
      }}
    ],
    "wants_to_reconnect": "Yes / Likely / No / Unclear",
    "reconnect_evidence": "Evidence or 'Not mentioned'"
  }},
  "service_request_signals": {{
    "detected": "Yes / No",
    "type": "Advisory / Paid Service / Partnership / Hiring / Investment / Not detected",
    "details": "Details or 'None detected'",
    "signals": [
      {{
        "signal": "description",
        "transcript_reference": "quote or 'Not available from transcript'"
      }}
    ]
  }},
  "venture_feedback": {{
    "overall_sentiment": "Very Positive / Positive / Neutral / Mixed / Negative",
    "rating_mentioned": "Rating given or 'Not mentioned'",
    "fit_assessment": "1-2 sentences on how well mentor fit the need.",
    "specific_praise": "What venture appreciated, or 'Not available'",
    "concerns_or_gaps": "Any concerns, or 'None mentioned'",
    "followup_requested": "Yes / No / Unclear"
  }},
  "overall_session_health": {{
    "score": "Score out of 10 as number only",
    "summary": "2-3 sentence overall assessment.",
    "recommendation": "Continue / Reconnect / Try Different Mentor / Needs Follow-up"
  }}
}}

Return ONLY the JSON object. No markdown, no extra text.
"""
                            try:
                                raw_response = call_ai(si_prompt, max_tokens=3000)
                                cleaned_response = re.sub(r"```json|```", "", raw_response).strip()
                                try:
                                    analysis = json.loads(cleaned_response)
                                except json.JSONDecodeError:
                                    match_obj = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                                    analysis = json.loads(match_obj.group()) if match_obj else None

                                new_record["analysis"] = analysis
                            except Exception as e:
                                st.error(f"Analysis error: {e}")

                    # Save the session
                    if selected_mentor not in st.session_state.si_sessions:
                        st.session_state.si_sessions[selected_mentor] = []
                    st.session_state.si_sessions[selected_mentor].append(new_record)
                    st.session_state.si_active_session_id = new_sid
                    st.session_state.si_view_mode = "view_session"
                    st.rerun()

# ==================== TAB 1: EXPERT SEARCH ====================
with tab_search:

    # ------------------ RENDER CHAT HISTORY ------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "recommendations":
                st.markdown(message["summary"])
                display_expert_results(message["recommendations"], df)
            elif message.get("type") == "expert_score":
                st.markdown(message["content"])
            elif message.get("type") == "retry_prompt":
                st.markdown(message["content"])
                if message == st.session_state.messages[-1] and st.session_state.pending_retry:
                    col_yes, col_no, col_gap = st.columns([1, 1, 4])
                    with col_yes:
                        if st.button("✅ Yes, retry", key="retry_yes"):
                            retry_q = st.session_state.retry_query
                            st.session_state.pending_retry = False
                            st.session_state.messages.append({"role": "user", "content": "Yes"})
                            with st.spinner("Retrying your search..."):
                                try:
                                    retry_results = run_search(retry_q, filtered_df, filtered_vectors)
                                    st.session_state.last_recommendations = retry_results
                                    st.session_state.last_query = retry_q
                                    t1 = len([r for r in retry_results if r.get("Tier") == "1"])
                                    t2 = len([r for r in retry_results if r.get("Tier") == "2"])
                                    save_to_history(retry_q, t1, t2)
                                    retry_summary = f"Found **{t1} Tier 1 expert(s)** and **{t2} Tier 2 expert(s)**."
                                    st.session_state.messages.append({
                                        "role": "assistant", "type": "recommendations",
                                        "summary": retry_summary, "recommendations": retry_results,
                                        "content": retry_summary
                                    })
                                except Exception as e:
                                    st.session_state.messages.append({
                                        "role": "assistant", "type": "text",
                                        "content": f"Sorry, encountered an error: {e}"
                                    })
                            st.rerun()
                    with col_no:
                        if st.button("❌ No, cancel", key="retry_no"):
                            st.session_state.pending_retry = False
                            st.session_state.retry_query = ""
                            st.session_state.messages.append({
                                "role": "assistant", "type": "text",
                                "content": "No problem! Feel free to try a new search anytime."
                            })
                            st.rerun()
            else:
                st.markdown(message["content"])

    # ------------------ HANDLE RE-RUN FROM HISTORY ------------------
    if st.session_state.get("_rerun_query"):
        user_input = st.session_state._rerun_query
        st.session_state._rerun_query = None
    else:
        user_input = None

    # ------------------ USER INPUT ------------------
    chat_input = st.chat_input(
        "Describe your business, ask a follow-up, or start a new search..."
    )
    if chat_input:
        user_input = chat_input

    # ------------------ PROCESS INPUT ------------------
    if user_input:

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        intent = detect_intent(user_input, st.session_state.last_recommendations)

        # -------- FOLLOWUP HANDLING --------
        if intent == "followup":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    conversation_history = ""
                    for msg in st.session_state.messages[-6:]:
                        role = "Founder" if msg["role"] == "user" else "Assistant"
                        content = msg.get("content", msg.get("summary", ""))
                        conversation_history += f"{role}: {content}\n"

                    followup_prompt = f"""
    You are an AI expert-matching assistant helping an Indian founder find the right expert.

    Original search query: "{st.session_state.last_query}"

    Previous conversation:
    {conversation_history}

    Current recommended experts (Tier 1 = Industry + Operator experience match, Tier 2 = Partial match):
    {json.dumps(st.session_state.last_recommendations, indent=2)}

    Founder's follow-up question: "{user_input}"

    Instructions:
    - Answer the follow-up question conversationally and helpfully
    - Always mention which Tier an expert belongs to when referencing them
    - If asked to compare experts, compare them clearly with pros and cons
    - If asked to refine search, explain what kind of expert would be better
    - If asked about a specific expert, give detailed insights
    - If asked for a different expert type, suggest what to look for
    - Always lead with Industry Match and Operator Experience when comparing
    - Reference expert names, designations, qualifications and scores where relevant
    - Keep response clear, structured and founder-friendly
    - Do NOT return JSON — return a natural conversational response
    """
                    try:
                        followup_response = call_ai(followup_prompt, max_tokens=1024)
                        st.markdown(followup_response)
                        st.session_state.messages.append({
                            "role": "assistant", "type": "text", "content": followup_response
                        })
                    except Exception as e:
                        st.error(f"AI Error: {e}")

        # -------- NEW SEARCH HANDLING --------
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for the best experts..."):
                    try:
                        ai_recommendations = run_search(user_input, filtered_df, filtered_vectors)

                        if not ai_recommendations:
                            st.session_state.pending_retry = True
                            st.session_state.retry_query = user_input
                            retry_msg = (
                                "🔍 Looks like I encountered a glitch and couldn't retrieve results. "
                                "Shall I try again?\n\n**Type Yes or No, or use the buttons below.**"
                            )
                            st.markdown(retry_msg)
                            col_yes, col_no, col_gap = st.columns([1, 1, 4])
                            with col_yes:
                                if st.button("✅ Yes, retry", key="inline_retry_yes"):
                                    retry_q = st.session_state.retry_query
                                    st.session_state.pending_retry = False
                                    st.session_state.messages.append({"role": "user", "content": "Yes"})
                                    with st.spinner("Retrying..."):
                                        try:
                                            retry_results = run_search(retry_q, filtered_df, filtered_vectors)
                                            st.session_state.last_recommendations = retry_results
                                            st.session_state.last_query = retry_q
                                            t1 = len([r for r in retry_results if r.get("Tier") == "1"])
                                            t2 = len([r for r in retry_results if r.get("Tier") == "2"])
                                            save_to_history(retry_q, t1, t2)
                                            retry_summary = f"Found **{t1} Tier 1 expert(s)** and **{t2} Tier 2 expert(s)**."
                                            st.session_state.messages.append({
                                                "role": "assistant", "type": "recommendations",
                                                "summary": retry_summary, "recommendations": retry_results,
                                                "content": retry_summary
                                            })
                                        except Exception as e:
                                            st.session_state.messages.append({
                                                "role": "assistant", "type": "text",
                                                "content": f"Sorry, encountered an error: {e}"
                                            })
                                    st.rerun()
                            with col_no:
                                if st.button("❌ No, cancel", key="inline_retry_no"):
                                    st.session_state.pending_retry = False
                                    st.session_state.retry_query = ""
                                    st.session_state.messages.append({
                                        "role": "assistant", "type": "text",
                                        "content": "No problem! Feel free to try a new search anytime."
                                    })
                                    st.rerun()
                            st.session_state.messages.append({
                                "role": "assistant", "type": "retry_prompt", "content": retry_msg
                            })

                        else:
                            st.session_state.last_recommendations = ai_recommendations
                            st.session_state.last_query = user_input
                            st.session_state.pending_retry = False

                            tier1_count = len([m for m in ai_recommendations if m.get("Tier") == "1"])
                            tier2_count = len([m for m in ai_recommendations if m.get("Tier") == "2"])

                            save_to_history(user_input, tier1_count, tier2_count)

                            prog_note = (
                                f"  *(filtered to: {', '.join(selected_programs)})*"
                                if selected_programs else ""
                            )
                            summary = (
                                f"Found **{tier1_count} Tier 1 expert(s)** "
                                f"(Industry + Operator experience match) and "
                                f"**{tier2_count} Tier 2 expert(s)** (partial match)"
                                f"{prog_note}.\n\n"
                                f"You can ask me to **compare any two experts**, "
                                f"**tell me more about a specific expert**, "
                                f"**refine the search**, or **start a new search** anytime."
                            )
                            st.session_state.messages.append({
                                "role": "assistant", "type": "recommendations",
                                "summary": summary, "recommendations": ai_recommendations,
                                "content": summary
                            })
                            # Rerun so sidebar re-renders with updated search history
                            st.rerun()

                    except Exception as e:
                        st.session_state.pending_retry = True
                        st.session_state.retry_query = user_input
                        retry_msg = (
                            "🔍 Looks like I encountered a glitch and couldn't retrieve results. "
                            "Shall I try again?\n\n**Type Yes or No, or use the buttons below.**"
                        )
                        st.markdown(retry_msg)
                        col_yes, col_no, col_gap = st.columns([1, 1, 4])
                        with col_yes:
                            if st.button("✅ Yes, retry", key="err_retry_yes"):
                                retry_q = st.session_state.retry_query
                                st.session_state.pending_retry = False
                                st.session_state.messages.append({"role": "user", "content": "Yes"})
                                with st.spinner("Retrying..."):
                                    try:
                                        retry_results = run_search(retry_q, filtered_df, filtered_vectors)
                                        st.session_state.last_recommendations = retry_results
                                        st.session_state.last_query = retry_q
                                        t1 = len([r for r in retry_results if r.get("Tier") == "1"])
                                        t2 = len([r for r in retry_results if r.get("Tier") == "2"])
                                        save_to_history(retry_q, t1, t2)
                                        retry_summary = f"Found **{t1} Tier 1 expert(s)** and **{t2} Tier 2 expert(s)**."
                                        st.session_state.messages.append({
                                            "role": "assistant", "type": "recommendations",
                                            "summary": retry_summary, "recommendations": retry_results,
                                            "content": retry_summary
                                        })
                                    except Exception as e:
                                        st.session_state.messages.append({
                                            "role": "assistant", "type": "text",
                                            "content": f"Sorry, encountered an error: {e}"
                                        })
                                st.rerun()
                        with col_no:
                            if st.button("❌ No, cancel", key="err_retry_no"):
                                st.session_state.pending_retry = False
                                st.session_state.retry_query = ""
                                st.session_state.messages.append({
                                    "role": "assistant", "type": "text",
                                    "content": "No problem! Feel free to try a new search anytime."
                                })
                                st.rerun()
                        st.session_state.messages.append({
                            "role": "assistant", "type": "retry_prompt", "content": retry_msg
                        })

        # -------- UPLOADED EXPERT PROFILE SCORING --------
        if expert_profile_text and st.session_state.last_recommendations:
            with st.chat_message("assistant"):
                with st.spinner("Scoring uploaded expert profile..."):

                    founder_context_section = ""
                    if founder_doc_text:
                        founder_context_section = (
                            f"Founder uploaded a business document:\n"
                            f"{founder_doc_text[:1500]}"
                        )

                    scoring_prompt = f"""
    A founder is looking for an expert with this requirement:
    "{st.session_state.last_query}"

    {founder_context_section}

    TIER RULES:
    - Tier 1: Expert matches BOTH industry AND has operator experience in founder's problem
    - Tier 2: Expert matches only one — either industry OR operator experience

    SCORING:
    - Industry Match: 3 points
    - Operator Experience: 3 points
    - Relevant Expertise: 2 points
    - Key Credentials: 2 points

    Current AI recommended experts for comparison:
    {json.dumps(st.session_state.last_recommendations, indent=2)}

    Evaluate this uploaded expert profile:
    {expert_profile_text[:2000]}

    Return strictly as a JSON object:
    {{
      "Uploaded Expert Name": "name if found, else 'Uploaded Expert'",
      "Tier": "1 or 2 based on tier rules above",
      "Tier Reason": "one line explaining why this expert is Tier 1 or Tier 2",
      "Overall Score": "score out of 10 as number only",
      "Industry Match Score": "score | one line explanation",
      "Hands On Score": "score | one line explanation",
      "Expertise Score": "score | one line explanation",
      "Credibility Score": "score | one line explanation — focus on industry recognition, awards, board memberships",
      "Hands On Experience": "Yes / No / Partial",
      "Hands On Details": "1-2 lines on what they have personally done as an operator",
      "Match Summary": "2-3 lines — lead with industry match and operator experience",
      "Key Strengths": "top 3 strengths relevant to founder's problem",
      "Gaps": "any gaps compared to what the founder needs",
      "Rank vs Tier 1": "how this expert compares to Tier 1 experts if any",
      "Rank vs Tier 2": "how this expert compares to Tier 2 experts"
    }}

    Return only the JSON object. No extra text.
    """
                    try:
                        score_raw = call_ai(scoring_prompt, max_tokens=1024)
                        score_cleaned = re.sub(r"```json|```", "", score_raw).strip()

                        try:
                            score_result = json.loads(score_cleaned)
                        except json.JSONDecodeError:
                            match_score = re.search(r'\{.*\}', score_cleaned, re.DOTALL)
                            score_result = (
                                json.loads(match_score.group()) if match_score else {}
                            )

                        if score_result:
                            expert_name = score_result.get("Uploaded Expert Name", "Uploaded Expert")
                            score_val   = score_result.get("Overall Score", "N/A")
                            hands_on_val = score_result.get("Hands On Experience", "").strip()
                            tier_val    = score_result.get("Tier", "2")
                            tier_reason = score_result.get("Tier Reason", "")

                            tier_color = "🏆" if tier_val == "1" else "🔍"
                            st.markdown(
                                f"---\n#### {tier_color} Uploaded Expert — {expert_name} | Tier {tier_val}"
                            )
                            if tier_val == "1":
                                st.success(f"✅ Tier 1 — {tier_reason}")
                            else:
                                st.warning(f"⚠️ Tier 2 — {tier_reason}")

                            # Program + experience badges for uploaded expert
                            render_program_badge(expert_name)
                            render_experience_badge(expert_name)

                            st.markdown("### 📊 Match Scorecard")
                            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                            with sc1:
                                st.metric("⭐ Overall", f"{score_val} / 10")
                            for col, key, label, max_val in [
                                (sc2, "Industry Match Score", "🏭 Industry", "3"),
                                (sc3, "Hands On Score", "🛠️ Hands-on/Operator Exp", "3"),
                                (sc4, "Expertise Score", "💼 Expertise", "2"),
                                (sc5, "Credibility Score", "🏅 Key Credentials", "2"),
                            ]:
                                with col:
                                    raw = score_result.get(key, "N/A")
                                    parts = (
                                        raw.split("|")
                                        if isinstance(raw, str) and "|" in raw
                                        else [raw, ""]
                                    )
                                    st.metric(label, f"{parts[0].strip()} / {max_val}")
                                    if len(parts) > 1:
                                        st.caption(parts[1].strip())

                            st.markdown("---")
                            st.markdown("**🛠️ Hands-on/Operator Experience**")
                            if hands_on_val == "Yes":
                                st.success(f"🟢 Yes — Hands-on/Operator — {score_result.get('Hands On Details', '')}")
                            elif hands_on_val == "Partial":
                                st.warning(f"🟡 Partial Hands-on/Operator — {score_result.get('Hands On Details', '')}")
                            else:
                                st.error(f"🔴 No Hands-on/Operator Experience — {score_result.get('Hands On Details', '')}")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**📊 Rank vs Tier 1**")
                                st.write(score_result.get("Rank vs Tier 1", "N/A"))
                            with col2:
                                st.markdown("**📊 Rank vs Tier 2**")
                                st.write(score_result.get("Rank vs Tier 2", "N/A"))

                            col3, col4 = st.columns(2)
                            with col3:
                                st.markdown("**✅ Key Strengths**")
                                st.write(score_result.get("Key Strengths", "N/A"))
                            with col4:
                                st.markdown("**⚠️ Gaps**")
                                st.write(score_result.get("Gaps", "N/A"))

                            st.markdown("**📝 Match Summary**")
                            st.write(score_result.get("Match Summary", "N/A"))

                            st.session_state.messages.append({
                                "role": "assistant", "type": "expert_score",
                                "content": (
                                    f"Uploaded expert **{expert_name}** is "
                                    f"**Tier {tier_val}** with score **{score_val}/10**. "
                                    f"{tier_reason}"
                                )
                            })

                    except Exception as e:
                        st.error(f"Expert Scoring Error: {e}")

        elif expert_profile_text and not st.session_state.last_recommendations:
            with st.chat_message("assistant"):
                st.info(
                    "💡 Expert profile uploaded. "
                    "Run a search first to get match analysis."
                )
