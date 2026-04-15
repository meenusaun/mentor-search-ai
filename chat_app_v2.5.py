import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
import os
import json
import re
import pdfplumber
import docx
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go

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
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.session_state.last_recommendations = []
    st.session_state.last_query = ""
    st.rerun()

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
        "Program", "Years of Experience", "Sector"
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

    df["combined"] = (
        "Expertise: " + df["Expertise"] + ". " +
        "Secondary Expertise: " + df["Secondary Expertise"] + ". " +
        "Industry: " + df["Industry"] + ". " +
        "Secondary Industry: " + df["Secondary Industry"] + ". " +
        "Description: " + df["Description"] + ". " +
        "Tags: " + df["Expertise Tags"] + " " + df["Industry Tags"] + ". " +
        "Qualification: " + df["Qualification"] + ". " +
        "Current Organization: " + df["Current Organization"] + ". " +
        "Current Designation: " + df["Current Designation"]
    )
    return df

df = load_data()

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ CREATE VECTORS ------------------
@st.cache_data
def get_vectors(texts):
    return model.encode(texts, batch_size=64, show_progress_bar=False)

vectors = get_vectors(df["combined"].tolist())

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

# ============================================================
# TAB LAYOUT — Search | Network Insights
# ============================================================
tab_search, tab_insights = st.tabs(["🔍 Expert Search", "📊 Network Insights"])

# ============================================================
# TAB 1 — EXPERT SEARCH
# ============================================================
with tab_search:

    # ------------------ PROGRAM FILTER ------------------
    st.markdown("### 🎯 Filter by Program")
    all_programs = sorted(
        set(
            p.strip()
            for progs in df["Program"].dropna()
            for p in str(progs).split(",")
            if p.strip() and p.strip() != ""
        )
    )

    selected_programs = st.multiselect(
        "Select Program(s) — leave empty to search all programs",
        options=all_programs,
        default=[],
        help="Filter experts to only those onboarded for specific programs"
    )

    # Apply program filter to working dataframe
    if selected_programs:
        mask = df["Program"].apply(
            lambda x: any(
                prog.strip() in [p.strip() for p in str(x).split(",")]
                for prog in selected_programs
            )
        )
        filtered_df = df[mask].reset_index(drop=True)
        filtered_vectors = get_vectors(filtered_df["combined"].tolist())
        st.info(
            f"🔎 Searching across **{len(filtered_df)} expert(s)** "
            f"onboarded for: **{', '.join(selected_programs)}**"
        )
    else:
        filtered_df = df
        filtered_vectors = vectors
        st.info(f"🔎 Searching across all **{len(filtered_df)} experts** in the network")

    st.markdown("---")

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

    # ------------------ DISPLAY SINGLE EXPERT CARD ------------------
    def display_expert_card(expert, index, tier_label, source_df):
        hands_on = expert.get("Hands On Experience", "").strip()
        if hands_on == "Yes":
            badge = "🟢 Hands-On/Operator"
        elif hands_on == "Partial":
            badge = "🟡 Partial Hands-on/Operator Experience"
        else:
            badge = "🔴 No Direct Experience"

        overall = expert.get("Overall Score", "N/A")

        with st.expander(
            f"#{index} — {expert.get('Name', 'N/A')} | "
            f"⭐ {overall}/10 | {badge}",
            expanded=(index == 1)
        ):
            # Program badge
            expert_name = expert.get("Name", "")
            program_val = ""
            if expert_name and "Name" in source_df.columns and "Program" in source_df.columns:
                match = source_df[source_df["Name"] == expert_name]
                if not match.empty:
                    program_val = match.iloc[0].get("Program", "")
            if program_val and str(program_val).strip():
                programs_list = [p.strip() for p in str(program_val).split(",") if p.strip()]
                badges_html = " ".join(
                    [f"<span style='background:#1F4E79;color:white;padding:2px 10px;"
                     f"border-radius:12px;font-size:12px;margin-right:4px;'>📌 {p}</span>"
                     for p in programs_list]
                )
                st.markdown(f"**Program:** {badges_html}", unsafe_allow_html=True)
                st.write("")

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

            linkedin_map = source_df.set_index("Name")["LinkedIn"].to_dict()
            linkedin = linkedin_map.get(expert.get("Name", ""), "")
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
- Always respond in the language and format specified in the user message""",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

    # ------------------ RENDER CHAT HISTORY ------------------
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message.get("type") == "recommendations":
                st.markdown(message["summary"])
                display_expert_results(message["recommendations"], filtered_df)
            elif message.get("type") == "expert_score":
                st.markdown(message["content"])
            else:
                st.markdown(message["content"])

    # ------------------ USER INPUT ------------------
    user_input = st.chat_input(
        "Describe your business, ask a follow-up, or start a new search..."
    )

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
                            "role": "assistant",
                            "type": "text",
                            "content": followup_response
                        })
                    except Exception as e:
                        st.error(f"AI Error: {e}")

        # -------- NEW SEARCH HANDLING --------
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for the best experts..."):

                    enriched_query = user_input
                    if founder_doc_text:
                        enriched_query = (
                            f"{user_input}\n\n"
                            f"Context from business document:\n{founder_doc_text[:1500]}"
                        )

                    query_vec = model.encode([enriched_query])
                    similarity = cosine_similarity(query_vec, filtered_vectors)
                    filtered_df["score"] = similarity[0]
                    candidates = filtered_df.sort_values(
                        by=["score", "Name"],
                        ascending=[False, True]
                    ).head(20)

                    # Program context for prompt
                    program_context = ""
                    if selected_programs:
                        program_context = (
                            f"\nIMPORTANT: The search is filtered to experts "
                            f"onboarded for program(s): {', '.join(selected_programs)}. "
                            f"Only recommend experts from this filtered set.\n"
                        )

                    expert_info = ""
                    for _, row in candidates.iterrows():
                        doc_summary = (
                            row["Doc Text"][:500]
                            if row["Doc Text"] and len(row["Doc Text"]) > 50
                            else "Not available"
                        )
                        program_info = row.get("Program", "")
                        expert_info += f"""
Name: {row['Name']}
Program(s): {program_info}
Expertise: {row['Expertise']}
Secondary Expertise: {row['Secondary Expertise']}
Industry: {row['Industry']}
Current Designation: {row['Current Designation']}
Current Organization: {row['Current Organization']}
Qualification: {row['Qualification']}
Description: {row['Description']}
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
                    if st.session_state.last_query:
                        previous_context = f"""
Note: The founder previously searched for: "{st.session_state.last_query}"
This is a new search. Treat it independently but keep previous context in mind.
"""

                    prompt = f"""
You are helping an Indian founder find the right expert.

Founder's business brief and problem statement:
"{user_input}"

{program_context}
{founder_context_section}
{previous_context}

Here are expert profiles to evaluate:
{expert_info}

TIER CLASSIFICATION RULES — This is the most important part:

TIER 1 — Strong Match (show minimum 1, maximum 5):
An expert qualifies for Tier 1 ONLY if BOTH conditions are true:
  ✅ Condition 1 — Industry Match: The expert has directly worked IN the same
     or very closely related industry as the founder's business. Not just advised
     — actually worked in it as an operator, founder, or senior leader.
  ✅ Condition 2 — Operator Experience: The expert has PERSONALLY done the
     specific task or solved the specific problem the founder is facing.
     Not consulting, not teaching, not advising — actually done it themselves
     on the ground as an operator.

If even ONE condition is missing → expert goes to Tier 2, NOT Tier 1.
Be strict. It is better to show 1 Tier 1 expert than to incorrectly
promote a weak match to Tier 1.

TIER 2 — Partial Match (maximum 5):
Experts who meet at least ONE of the following:
  - Matches the industry but lacks operator experience in the specific problem
  - Has operator experience in the problem area but from a different industry
  - Has strong relevant expertise that could still be useful to the founder

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
Total array can have between 2 and 10 objects.

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
    "Hands On Details": "1-2 lines on what they have personally done as an operator. If No/Partial, state clearly what is missing."
  }}
]

Return only the JSON array. No extra text, no markdown outside the array.
"""

                    try:
                        ai_raw = call_ai(prompt, max_tokens=3000)

                        cleaned = re.sub(r"```json|```", "", ai_raw).strip()
                        try:
                            ai_recommendations = json.loads(cleaned)
                        except json.JSONDecodeError:
                            match_json = re.search(r'\[.*\]', cleaned, re.DOTALL)
                            ai_recommendations = (
                                json.loads(match_json.group()) if match_json else []
                            )

                        if ai_recommendations:
                            st.session_state.last_recommendations = ai_recommendations
                            st.session_state.last_query = user_input

                            tier1_count = len(
                                [m for m in ai_recommendations if m.get("Tier") == "1"]
                            )
                            tier2_count = len(
                                [m for m in ai_recommendations if m.get("Tier") == "2"]
                            )

                            program_note = ""
                            if selected_programs:
                                program_note = (
                                    f" *(filtered to: {', '.join(selected_programs)})*"
                                )

                            summary = (
                                f"Found **{tier1_count} Tier 1 expert(s)** "
                                f"(Industry + Operator experience match) and "
                                f"**{tier2_count} Tier 2 expert(s)** (partial match)"
                                f"{program_note}.\n\n"
                                f"You can ask me to **compare any two experts**, "
                                f"**tell me more about a specific expert**, "
                                f"**refine the search**, or **start a new search** anytime."
                            )
                            st.markdown(summary)
                            display_expert_results(ai_recommendations, filtered_df)

                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "recommendations",
                                "summary": summary,
                                "recommendations": ai_recommendations,
                                "content": summary
                            })
                        else:
                            fallback = (
                                "I could not find strong matches. "
                                "Could you describe your business problem in more detail?"
                            )
                            st.markdown(fallback)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "text",
                                "content": fallback
                            })

                    except Exception as e:
                        st.error(f"AI Error: {e}")

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
                            expert_name = score_result.get(
                                "Uploaded Expert Name", "Uploaded Expert"
                            )
                            score_val = score_result.get("Overall Score", "N/A")
                            hands_on_val = score_result.get(
                                "Hands On Experience", ""
                            ).strip()
                            tier_val = score_result.get("Tier", "2")
                            tier_reason = score_result.get("Tier Reason", "")

                            tier_color = "🏆" if tier_val == "1" else "🔍"
                            st.markdown(
                                f"---\n#### {tier_color} Uploaded Expert — "
                                f"{expert_name} | Tier {tier_val}"
                            )

                            if tier_val == "1":
                                st.success(f"✅ Tier 1 — {tier_reason}")
                            else:
                                st.warning(f"⚠️ Tier 2 — {tier_reason}")

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
                                st.success(
                                    f"🟢 Yes — Hands-on/Operator — {score_result.get('Hands On Details', '')}"
                                )
                            elif hands_on_val == "Partial":
                                st.warning(
                                    f"🟡 Partial Hands-on/Operator — "
                                    f"{score_result.get('Hands On Details', '')}"
                                )
                            else:
                                st.error(
                                    f"🔴 No Hands-on/Operator Experience — {score_result.get('Hands On Details', '')}"
                                )

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

                            score_content = (
                                f"Uploaded expert **{expert_name}** is "
                                f"**Tier {tier_val}** with score **{score_val}/10**. "
                                f"{tier_reason}"
                            )
                            st.session_state.messages.append({
                                "role": "assistant",
                                "type": "expert_score",
                                "content": score_content
                            })

                    except Exception as e:
                        st.error(f"Expert Scoring Error: {e}")

        elif expert_profile_text and not st.session_state.last_recommendations:
            with st.chat_message("assistant"):
                st.info(
                    "💡 Expert profile uploaded. "
                    "Run a search first to get match analysis."
                )


# ============================================================
# TAB 2 — NETWORK INSIGHTS
# ============================================================
with tab_insights:

    st.markdown("## 📊 Network Insights")
    st.markdown("An overview of the expert network across programs, experience, expertise, and sectors.")
    st.markdown("---")

    # ── Insight filter by program ──
    insight_programs = st.multiselect(
        "Filter insights by Program (leave empty for full network view)",
        options=all_programs,
        default=[],
        key="insight_program_filter"
    )

    if insight_programs:
        ins_mask = df["Program"].apply(
            lambda x: any(
                prog.strip() in [p.strip() for p in str(x).split(",")]
                for prog in insight_programs
            )
        )
        ins_df = df[ins_mask].reset_index(drop=True)
    else:
        ins_df = df.copy()

    total = len(ins_df)
    prog_label = ", ".join(insight_programs) if insight_programs else "All Programs"
    st.markdown(f"**Showing insights for: {prog_label} — {total} expert(s)**")
    st.markdown("---")

    # ── Summary metrics ──
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("👥 Total Experts", total)
    with m2:
        num_programs = len(all_programs)
        st.metric("🎯 Total Programs", num_programs)
    with m3:
        unique_industries = ins_df["Industry"].replace("", pd.NA).dropna().nunique()
        st.metric("🏭 Unique Industries", unique_industries)
    with m4:
        unique_expertise = ins_df["Expertise"].replace("", pd.NA).dropna().nunique()
        st.metric("💼 Unique Expertise Areas", unique_expertise)

    st.markdown("---")

    # ── Chart 1: Mentors by Program ──
    st.subheader("1️⃣  Mentors by Program")

    program_rows = []
    for _, row in ins_df.iterrows():
        progs = [p.strip() for p in str(row["Program"]).split(",") if p.strip()]
        for prog in progs:
            program_rows.append({"Program": prog, "Name": row["Name"]})

    if program_rows:
        prog_df = pd.DataFrame(program_rows)
        prog_count = prog_df.groupby("Program").size().reset_index(name="Count")
        prog_count = prog_count.sort_values("Count", ascending=False)

        fig1 = px.bar(
            prog_count,
            x="Program",
            y="Count",
            text="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Number of Experts per Program"
        )
        fig1.update_traces(textposition="outside")
        fig1.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            xaxis_title="Program",
            yaxis_title="Number of Experts",
            plot_bgcolor="white"
        )
        st.plotly_chart(fig1, use_container_width=True)

        with st.expander("📋 View Program-wise Expert List"):
            for prog in prog_count["Program"].tolist():
                names = prog_df[prog_df["Program"] == prog]["Name"].tolist()
                st.markdown(f"**{prog}** ({len(names)} experts)")
                st.write(", ".join(names))
    else:
        st.info("No Program data available. Add a 'Program' column to your Excel file.")

    st.markdown("---")

    # ── Chart 2: Mentors by Experience ──
    st.subheader("2️⃣  Mentors by Years of Experience")

    if "Years of Experience" in ins_df.columns and ins_df["Years of Experience"].replace("", pd.NA).dropna().shape[0] > 0:
        exp_df = ins_df[ins_df["Years of Experience"].str.strip() != ""].copy()

        def bucket_experience(val):
            try:
                yrs = int(str(val).strip().split()[0])
                if yrs < 5:
                    return "0–5 years"
                elif yrs < 10:
                    return "5–10 years"
                elif yrs < 15:
                    return "10–15 years"
                elif yrs < 20:
                    return "15–20 years"
                elif yrs < 25:
                    return "20–25 years"
                else:
                    return "25+ years"
            except Exception:
                return "Not Specified"

        exp_df["Exp Bucket"] = exp_df["Years of Experience"].apply(bucket_experience)
        exp_count = exp_df["Exp Bucket"].value_counts().reset_index()
        exp_count.columns = ["Experience Band", "Count"]

        order = ["0–5 years", "5–10 years", "10–15 years", "15–20 years", "20–25 years", "25+ years", "Not Specified"]
        exp_count["sort_key"] = exp_count["Experience Band"].apply(
            lambda x: order.index(x) if x in order else 99
        )
        exp_count = exp_count.sort_values("sort_key").drop(columns="sort_key")

        fig2 = px.pie(
            exp_count,
            names="Experience Band",
            values="Count",
            title="Mentor Distribution by Years of Experience",
            color_discrete_sequence=px.colors.sequential.Blues_r
        )
        fig2.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        # Fallback: use Description length as proxy
        st.info(
            "💡 No 'Years of Experience' column found. "
            "Add this column to your Excel for accurate experience insights. "
            "Showing a placeholder chart."
        )
        placeholder_data = pd.DataFrame({
            "Experience Band": ["0–5 years", "5–10 years", "10–15 years", "15–20 years", "20–25 years", "25+ years"],
            "Count": [0, 0, 0, 0, 0, 0]
        })
        st.dataframe(placeholder_data)

    st.markdown("---")

    # ── Chart 3: Mentors by Expertise ──
    st.subheader("3️⃣  Mentors by Expertise")

    expertise_rows = []
    for _, row in ins_df.iterrows():
        for col in ["Expertise", "Secondary Expertise"]:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                for item in val.split(","):
                    item = item.strip()
                    if item:
                        expertise_rows.append(item)

    if expertise_rows:
        exp_series = pd.Series(expertise_rows)
        top_expertise = exp_series.value_counts().head(20).reset_index()
        top_expertise.columns = ["Expertise Area", "Count"]

        fig3 = px.bar(
            top_expertise,
            x="Count",
            y="Expertise Area",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Teal",
            title="Top 20 Expertise Areas in the Network"
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Number of Experts",
            yaxis_title="",
            plot_bgcolor="white",
            height=600
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No expertise data available.")

    st.markdown("---")

    # ── Chart 4: Mentors by Sector ──
    st.subheader("4️⃣  Mentors by Sector")

    sector_col = "Sector" if "Sector" in ins_df.columns else "Industry"

    sector_rows = []
    for _, row in ins_df.iterrows():
        for col in [sector_col, "Secondary Industry"]:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                for item in val.split(","):
                    item = item.strip()
                    if item:
                        sector_rows.append(item)

    if sector_rows:
        sec_series = pd.Series(sector_rows)
        top_sectors = sec_series.value_counts().head(20).reset_index()
        top_sectors.columns = ["Sector", "Count"]

        fig4 = px.treemap(
            top_sectors,
            path=["Sector"],
            values="Count",
            title="Mentor Coverage by Sector (Treemap)",
            color="Count",
            color_continuous_scale="Blues"
        )
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)

        # Also show as horizontal bar
        fig4b = px.bar(
            top_sectors,
            x="Count",
            y="Sector",
            orientation="h",
            text="Count",
            color="Count",
            color_continuous_scale="Blues",
            title="Top 20 Sectors — Mentor Count"
        )
        fig4b.update_traces(textposition="outside")
        fig4b.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            yaxis=dict(autorange="reversed"),
            xaxis_title="Number of Experts",
            yaxis_title="",
            plot_bgcolor="white",
            height=600
        )
        st.plotly_chart(fig4b, use_container_width=True)
    else:
        st.info("No sector/industry data available.")

    st.markdown("---")

    # ── Full Expert Table ──
    with st.expander("📋 View Full Expert Directory"):
        display_cols = [c for c in [
            "Name", "Program", "Expertise", "Industry",
            "Current Designation", "Current Organization",
            "Years of Experience", "Sector", "LinkedIn"
        ] if c in ins_df.columns]
        st.dataframe(
            ins_df[display_cols].replace("", "—"),
            use_container_width=True
        )
