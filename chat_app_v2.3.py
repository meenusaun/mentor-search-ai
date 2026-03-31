#--------------READING LINKEDIN PROFILE PDFs---------------------------
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

# ------------------ CLIENTS ------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
openai_client = OpenAI()
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Resources Network - Look for Mentor",
    page_icon="🔍",
    layout="wide"
)

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.image("DP_BG1.png", width=150)
st.write("")
with col2:
    st.markdown(
        "<h2 style='text-align: center;'>🌐 Resources Network - Look for Mentor</h2>",
        unsafe_allow_html=True
    )

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚙️ Settings")
ai_model = st.sidebar.radio(
    "Choose AI Model for Recommendations:",
    options=["GPT-4o Mini (OpenAI)", "Claude (Anthropic)"],
    index=0
)
st.sidebar.markdown("---")
if ai_model == "GPT-4o Mini (OpenAI)":
    st.sidebar.info("Using **OpenAI GPT-4o Mini** for recommendations.")
else:
    st.sidebar.info("Using **Anthropic Claude** for recommendations.")

# ------------------ FILE UPLOADERS ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Upload Business Document")
st.sidebar.caption("Upload your pitch deck or business plan to improve mentor matching.")
founder_uploaded_file = st.sidebar.file_uploader(
    "Pitch Deck / Business Document",
    type=["pdf", "docx", "txt"],
    key="founder_doc"
)

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Check a Mentor's Profile")
st.sidebar.caption("Upload a mentor's resume to score them against your requirement.")
mentor_uploaded_file = st.sidebar.file_uploader(
    "Mentor Resume / Profile",
    type=["pdf", "docx", "txt"],
    key="mentor_profile"
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
    except Exception as e:
        text = f"[PDF extraction error: {e}]"
    return text.strip()

def extract_text_from_docx_bytes(file_bytes):
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = f"[DOCX extraction error: {e}]"
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
    file_path = os.path.normpath(file_path)

    if not os.path.exists(file_path):
        relative = os.path.join(os.getcwd(), file_path)
        if os.path.exists(relative):
            file_path = relative
        else:
            return ""

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
    except Exception as e:
        return ""
    return ""

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl")

    for col in ["Expertise", "Secondary Expertise", "Industry", "Secondary Industry",
                "Description", "Expertise Tags", "Industry Tags"]:
        df[col] = df[col].fillna("").astype(str)

    for col in ["Document Path", "LinkedIn", "Qualification",
                "Current Organization", "Current Designation"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["Doc Text"] = df["Document Path"].apply(extract_text_from_file_path)

    df["PDF Available"] = df["Doc Text"].apply(
        lambda x: "✅ Yes" if x and len(x) > 50 else "❌ No"
    )

    df["combined"] = (
        "Expertise: " + df["Expertise"] + ". " +
        "Secondary Expertise: " + df["Secondary Expertise"] + ". " +
        "Industry: " + df["Industry"] + ". " +
        "Secondary Industry: " + df["Secondary Industry"] + ". " +
        "Description: " + df["Description"] + ". " +
        "Tags: " + df["Expertise Tags"] + " " + df["Industry Tags"] + ". " +
        "Qualification: " + df["Qualification"] + ". " +
        "Current Organization: " + df["Current Organization"] + ". " +
        "Current Designation: " + df["Current Designation"] + ". " +
        "Document: " + df["Doc Text"].str[:1000]
    )
    return df

df = load_data()

# ------------------ PDF LOAD STATUS IN SIDEBAR ------------------
total_mentors = len(df)
pdf_loaded = len(df[df["PDF Available"] == "✅ Yes"])
st.sidebar.markdown("---")
st.sidebar.subheader("📊 PDF Profile Status")
st.sidebar.progress(pdf_loaded / total_mentors if total_mentors > 0 else 0)
st.sidebar.caption(f"✅ {pdf_loaded} of {total_mentors} mentor PDFs loaded")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ CREATE VECTORS ------------------
@st.cache_data
def get_vectors(texts):
    return model.encode(texts)

vectors = get_vectors(df["combined"])

# ------------------ EXTRACT UPLOADED FILE TEXTS ------------------
founder_doc_text = ""
if founder_uploaded_file:
    founder_doc_text = extract_text_from_uploaded_file(founder_uploaded_file)
    if founder_doc_text:
        st.sidebar.success("✅ Business document parsed!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from business document.")

mentor_profile_text = ""
if mentor_uploaded_file:
    mentor_profile_text = extract_text_from_uploaded_file(mentor_uploaded_file)
    if mentor_profile_text:
        st.sidebar.success("✅ Mentor profile parsed!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from mentor profile.")

# ------------------ SESSION STATE INIT ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_recommendations" not in st.session_state:
    st.session_state.last_recommendations = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# ------------------ INTENT DETECTION ------------------
def detect_intent(user_input, last_recommendations):
    followup_keywords = [
        "tell me more", "more about", "compare", "vs", "versus",
        "which is better", "difference between", "what about",
        "explain", "elaborate", "details about", "why is", "how is",
        "refine", "show more", "different", "another", "instead",
        "same industry", "similar", "like #", "mentor #", "first mentor",
        "second mentor", "third mentor", "top mentor", "number"
    ]
    input_lower = user_input.lower()
    has_recommendations = len(last_recommendations) > 0
    is_followup = has_recommendations and any(
        kw in input_lower for kw in followup_keywords
    )
    return "followup" if is_followup else "new_search"

# ------------------ DISPLAY SINGLE MENTOR CARD ------------------
def display_mentor_card(mentor, index, df):
    hands_on = mentor.get("Hands On Experience", "").strip()
    if hands_on == "Yes":
        badge = "🟢 Hands-On"
    elif hands_on == "Partial":
        badge = "🟡 Partial"
    else:
        badge = "🔴 No Direct Experience"

    overall = mentor.get("Overall Score", "N/A")

    with st.expander(
        f"#{index} — {mentor.get('Name', 'N/A')} | "
        f"{mentor.get('Current Designation', '')} at "
        f"{mentor.get('Current Organization', '')} | "
        f"⭐ {overall}/10 | {badge}",
        expanded=(index == 1)
    ):
        # ---- WEIGHTED SCORECARD ----
        st.markdown("### 📊 Match Scorecard")
        sc1, sc2, sc3, sc4 = st.columns(4)

        for col, key, label, max_val in [
            (sc1, "Industry Match Score", "🏭 Industry Match", "3"),
            (sc2, "Hands On Score", "🛠️ Hands-On Exp", "3"),
            (sc3, "Expertise Score", "💼 Expertise", "2"),
            (sc4, "Credibility Score", "🎓 Credibility", "2"),
        ]:
            with col:
                raw = mentor.get(key, "N/A")
                parts = (
                    raw.split("|")
                    if isinstance(raw, str) and "|" in raw
                    else [raw, ""]
                )
                st.metric(label, f"{parts[0].strip()} / {max_val}")
                if len(parts) > 1:
                    st.caption(parts[1].strip())

        st.markdown("---")

        # ---- PROFILE INFO ----
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🎓 Qualification**")
            st.write(mentor.get("Qualification", "Not available"))
        with col2:
            st.markdown("**🏢 Currently Working At**")
            st.write(
                f"{mentor.get('Current Designation', '')} — "
                f"{mentor.get('Current Organization', '')}"
            )

        st.markdown("**✅ Why Suitable**")
        st.write(mentor.get("Match Reason", ""))

        st.markdown("**💼 Relevant Experience**")
        st.write(mentor.get("Relevant Experience", ""))

        st.markdown("**🛠️ Hands-On Experience in Founder's Area**")
        if hands_on == "Yes":
            st.success(f"✅ Yes — {mentor.get('Hands On Details', '')}")
        elif hands_on == "Partial":
            st.warning(f"⚠️ Partial — {mentor.get('Hands On Details', '')}")
        else:
            st.error(f"❌ No Direct Experience — {mentor.get('Hands On Details', '')}")

        # PDF + LinkedIn
        matched_row = df[df["Name"] == mentor.get("Name")]
        if not matched_row.empty:
            pdf_status = matched_row.iloc[0].get("PDF Available", "❌ No")
            st.caption(f"📄 PDF Profile: {pdf_status}")
            linkedin = matched_row.iloc[0].get("LinkedIn", "")
            if pd.notna(linkedin) and str(linkedin).strip() != "":
                st.markdown(f"[🔗 View LinkedIn Profile]({linkedin})")

# ------------------ DISPLAY FULL RESULTS ------------------
def display_mentor_results(ai_recommendations, df):
    if not ai_recommendations:
        return

    tier1 = [m for m in ai_recommendations if m.get("Tier") == "1"]
    tier2 = [m for m in ai_recommendations if m.get("Tier") == "2"]

    # ---- TIER 1 ----
    if tier1:
        st.markdown("""
        ## 🏆 Tier 1 — Strong Matches
        > These mentors match **both the industry AND have hands-on experience**
        in the founder's problem area.
        """)
        for i, mentor in enumerate(tier1):
            display_mentor_card(mentor, i + 1, df)
    else:
        st.warning(
            "⚠️ No Tier 1 matches found — no mentor matched both industry "
            "AND hands-on experience for this requirement."
        )

    # ---- TIER 2 ----
    if tier2:
        st.markdown("---")
        st.markdown("""
        ## 🔍 Tier 2 — Partial Matches
        > These mentors match **either the industry OR have relevant experience**
        but not both. They may still provide useful guidance.
        """)
        for i, mentor in enumerate(tier2):
            display_mentor_card(mentor, i + 1, df)

# ------------------ AI CALL HELPER ------------------
def call_ai(prompt, max_tokens=2048):
    if ai_model == "GPT-4o Mini (OpenAI)":
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    else:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

# ------------------ RENDER CHAT HISTORY ------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "recommendations":
            st.markdown(message["summary"])
            display_mentor_results(message["recommendations"], df)
        elif message.get("type") == "mentor_score":
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
You are an AI mentor-matching assistant helping an Indian founder find the right mentor.

Original search query: "{st.session_state.last_query}"

Previous conversation:
{conversation_history}

Current recommended mentors (Tier 1 = Industry + Hands-On match, Tier 2 = Partial match):
{json.dumps(st.session_state.last_recommendations, indent=2)}

Founder's follow-up question: "{user_input}"

Instructions:
- Answer the follow-up question conversationally and helpfully
- Always mention which Tier a mentor belongs to when referencing them
- If asked to compare mentors, compare them clearly with pros and cons
- If asked to refine search, explain what kind of mentor would be better
- If asked about a specific mentor, give detailed insights
- If asked for a different mentor type, suggest what to look for
- Always lead with Industry Match and Hands-On Experience when comparing
- Reference mentor names, designations, qualifications and scores where relevant
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
            with st.spinner("Searching for the best mentors..."):

                enriched_query = user_input
                if founder_doc_text:
                    enriched_query = (
                        f"{user_input}\n\n"
                        f"Context from business document:\n{founder_doc_text[:1500]}"
                    )

                query_vec = model.encode([enriched_query])
                similarity = cosine_similarity(query_vec, vectors)
                df["score"] = similarity[0]
                candidates = df.sort_values(by="score", ascending=False).head(15)

                mentor_info = ""
                for _, row in candidates.iterrows():
                    doc_summary = (
                        row["Doc Text"][:500]
                        if row["Doc Text"] and len(row["Doc Text"]) > 50
                        else "Not available"
                    )
                    mentor_info += f"""
Name: {row['Name']}
Expertise: {row['Expertise']}
Secondary Expertise: {row['Secondary Expertise']}
Industry: {row['Industry']}
Current Designation: {row['Current Designation']}
Current Organization: {row['Current Organization']}
Qualification: {row['Qualification']}
Description: {row['Description']}
PDF Profile Summary: {doc_summary}
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
You are helping an Indian founder find the right mentor.

Founder's business brief and problem statement:
"{user_input}"

{founder_context_section}
{previous_context}

Here are mentor profiles to evaluate:
{mentor_info}

TIER CLASSIFICATION RULES — This is the most important part:

TIER 1 — Strong Match (minimum 1, maximum 5):
A mentor qualifies for Tier 1 ONLY if BOTH conditions are true:
  ✅ Condition 1 — Industry Match: The mentor has directly worked IN the same
     or very closely related industry as the founder's business. Not just advised
     — actually worked in it as an operator, founder, or senior leader.
  ✅ Condition 2 — Hands-On Experience: The mentor has PERSONALLY done the
     specific task or solved the specific problem the founder is facing.
     Not consulting, not teaching — actually done it themselves on the ground.

If even ONE condition is missing → mentor goes to Tier 2, NOT Tier 1.
Be strict. It is better to show 1 Tier 1 mentor than incorrectly promote
a weak match to Tier 1.

TIER 2 — Partial Match (maximum 5):
Mentors who meet at least ONE of the following:
  - Matches the industry but lacks hands-on experience in the specific problem
  - Has hands-on experience in the problem but from a different industry
  - Has strong relevant expertise that could still be useful

SCORING (apply to all mentors):
- Industry Match: 3 points
- Hands-On Experience: 3 points
- Relevant Expertise: 2 points
- Qualification + Credibility: 2 points

STRICT RULES:
- Never put a mentor in Tier 1 just because they are impressive or well-qualified
- Industry match alone is NOT enough for Tier 1
- Hands-on alone is NOT enough for Tier 1
- BOTH must be present for Tier 1
- Hands On Experience: Yes = personally done it,
  Partial = advised/consulted on it, No = no relevant experience
- If PDF Profile Summary is available, use it to validate industry and hands-on claims

Return strictly as a JSON array.
Include ALL evaluated mentors — Tier 1 first (1-5), then Tier 2 (up to 5).
Total between 2 and 10 objects.

Format:
[
  {{
    "Tier": "1",
    "Name": "mentor name exactly as given",
    "Overall Score": "score out of 10 as number only e.g. 8",
    "Industry Match Score": "score | explanation e.g. 3 | Worked in manufacturing exports for 10 years",
    "Hands On Score": "score | explanation e.g. 3 | Personally handled DGFT and LC documentation",
    "Expertise Score": "score | explanation e.g. 2 | Strong supply chain expertise",
    "Credibility Score": "score | explanation e.g. 1 | MBA from regional institute",
    "Match Reason": "2-3 lines — lead with WHY they qualify for this tier",
    "Relevant Experience": "specific experience directly relevant to the founder's problem",
    "Current Designation": "their current designation",
    "Current Organization": "their current organization",
    "Qualification": "their qualification",
    "Hands On Experience": "Yes / No / Partial",
    "Hands On Details": "1-2 lines on what they personally did. If No/Partial, state what is missing."
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

                        summary = (
                            f"Found **{tier1_count} Tier 1 mentor(s)** "
                            f"(Industry + Hands-On match) and "
                            f"**{tier2_count} Tier 2 mentor(s)** (partial match).\n\n"
                            f"You can ask me to **compare any two mentors**, "
                            f"**tell me more about a specific mentor**, "
                            f"**refine the search**, or **start a new search** anytime."
                        )
                        st.markdown(summary)
                        display_mentor_results(ai_recommendations, df)

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

    # -------- UPLOADED MENTOR PROFILE SCORING --------
    if mentor_profile_text and st.session_state.last_recommendations:
        with st.chat_message("assistant"):
            with st.spinner("Scoring uploaded mentor profile..."):

                founder_context_section = ""
                if founder_doc_text:
                    founder_context_section = (
                        f"Founder uploaded a business document:\n"
                        f"{founder_doc_text[:1500]}"
                    )

                scoring_prompt = f"""
A founder is looking for a mentor with this requirement:
"{st.session_state.last_query}"

{founder_context_section}

TIER RULES:
- Tier 1: Mentor matches BOTH industry AND hands-on experience in founder's problem
- Tier 2: Mentor matches only one — either industry OR hands-on experience

SCORING:
- Industry Match: 3 points
- Hands-On Experience: 3 points
- Relevant Expertise: 2 points
- Qualification + Credibility: 2 points

Current AI recommended mentors for comparison:
{json.dumps(st.session_state.last_recommendations, indent=2)}

Evaluate this uploaded mentor profile:
{mentor_profile_text[:2000]}

Return strictly as a JSON object:
{{
  "Uploaded Mentor Name": "name if found, else 'Uploaded Mentor'",
  "Tier": "1 or 2",
  "Tier Reason": "one line explaining why Tier 1 or Tier 2",
  "Overall Score": "score out of 10 as number only",
  "Industry Match Score": "score | explanation",
  "Hands On Score": "score | explanation",
  "Expertise Score": "score | explanation",
  "Credibility Score": "score | explanation",
  "Hands On Experience": "Yes / No / Partial",
  "Hands On Details": "1-2 lines on what they personally did",
  "Match Summary": "2-3 lines leading with industry and hands-on",
  "Key Strengths": "top 3 strengths relevant to founder's problem",
  "Gaps": "any gaps compared to what the founder needs",
  "Rank vs Tier 1": "how this mentor compares to Tier 1 mentors",
  "Rank vs Tier 2": "how this mentor compares to Tier 2 mentors"
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
                        mentor_name = score_result.get(
                            "Uploaded Mentor Name", "Uploaded Mentor"
                        )
                        score_val = score_result.get("Overall Score", "N/A")
                        hands_on_val = score_result.get(
                            "Hands On Experience", ""
                        ).strip()
                        tier_val = score_result.get("Tier", "2")
                        tier_reason = score_result.get("Tier Reason", "")

                        tier_color = "🏆" if tier_val == "1" else "🔍"
                        st.markdown(
                            f"---\n#### {tier_color} Uploaded Mentor — "
                            f"{mentor_name} | Tier {tier_val}"
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
                            (sc3, "Hands On Score", "🛠️ Hands-On", "3"),
                            (sc4, "Expertise Score", "💼 Expertise", "2"),
                            (sc5, "Credibility Score", "🎓 Credibility", "2"),
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

                        st.markdown("**🛠️ Hands-On Experience**")
                        if hands_on_val == "Yes":
                            st.success(
                                f"🟢 Yes — {score_result.get('Hands On Details', '')}"
                            )
                        elif hands_on_val == "Partial":
                            st.warning(
                                f"🟡 Partial — "
                                f"{score_result.get('Hands On Details', '')}"
                            )
                        else:
                            st.error(
                                f"🔴 No — {score_result.get('Hands On Details', '')}"
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
                            f"Uploaded mentor **{mentor_name}** is "
                            f"**Tier {tier_val}** with score **{score_val}/10**. "
                            f"{tier_reason}"
                        )
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "mentor_score",
                            "content": score_content
                        })

                except Exception as e:
                    st.error(f"Mentor Scoring Error: {e}")

    elif mentor_profile_text and not st.session_state.last_recommendations:
        with st.chat_message("assistant"):
            st.info(
                "💡 Mentor profile uploaded. "
                "Run a search first to get match analysis."
            )