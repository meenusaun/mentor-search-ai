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
        return f"[File read error: {e}]"
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
    is_followup = has_recommendations and any(kw in input_lower for kw in followup_keywords)
    return "followup" if is_followup else "new_search"

# ------------------ DISPLAY SINGLE MENTOR CARD ------------------
def display_mentor_card(mentor, index, tier_label, df):
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

        with sc1:
            industry_score = mentor.get("Industry Match Score", "N/A")
            parts = industry_score.split("|") if isinstance(industry_score, str) and "|" in industry_score else [industry_score, ""]
            st.metric("🏭 Industry Match", f"{parts[0].strip()} / 3")
            if len(parts) > 1:
                st.caption(parts[1].strip())

        with sc2:
            hands_on_score = mentor.get("Hands On Score", "N/A")
            parts = hands_on_score.split("|") if isinstance(hands_on_score, str) and "|" in hands_on_score else [hands_on_score, ""]
            st.metric("🛠️ Hands-On Exp", f"{parts[0].strip()} / 3")
            if len(parts) > 1:
                st.caption(parts[1].strip())

        with sc3:
            expertise_score = mentor.get("Expertise Score", "N/A")
            parts = expertise_score.split("|") if isinstance(expertise_score, str) and "|" in expertise_score else [expertise_score, ""]
            st.metric("💼 Expertise", f"{parts[0].strip()} / 2")
            if len(parts) > 1:
                st.caption(parts[1].strip())

        with sc4:
            cred_score = mentor.get("Credibility Score", "N/A")
            parts = cred_score.split("|") if isinstance(cred_score, str) and "|" in cred_score else [cred_score, ""]
            st.metric("🎓 Credibility", f"{parts[0].strip()} / 2")
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

        matched_row = df[df["Name"] == mentor.get("Name")]
        if not matched_row.empty:
            linkedin = matched_row.iloc[0].get("LinkedIn", "")
            if pd.notna(linkedin) and str(linkedin).strip() != "":
                st.markdown(f"[🔗 View LinkedIn Profile]({linkedin})")


# ------------------ DISPLAY TABLE ------------------
def display_table(mentors_list, df, title):
    st.subheader(title)
    if not mentors_list:
        st.info("No mentors in this category.")
        return

    ai_df = pd.DataFrame(mentors_list)

    linkedin_map = df.set_index("Name")["LinkedIn"].to_dict()
    ai_df["LinkedIn Profile"] = ai_df["Name"].map(linkedin_map).fillna("")

    def make_clickable(link):
        if pd.notna(link) and str(link).strip() != "":
            return f'<a href="{link}" target="_blank">View Profile</a>'
        return "Not Available"

    ai_df["LinkedIn Profile"] = ai_df["LinkedIn Profile"].apply(make_clickable)

    score_map = df.set_index("Name")["score"].to_dict() if "score" in df.columns else {}
    ai_df["Embedding Score"] = ai_df["Name"].map(score_map).fillna(0).round(2)

    desc_map = df.set_index("Name")["Description"].to_dict()
    ai_df["Short Description"] = ai_df["Name"].map(desc_map).apply(
        lambda x: (x[:100] + "...") if isinstance(x, str) and len(x) > 100 else x
    )

    industry_map = df.set_index("Name")["Industry"].to_dict()
    ai_df["Industry"] = ai_df["Name"].map(industry_map).fillna("")

    def color_hands_on(val):
        if val == "Yes":
            return "🟢 Yes"
        elif val == "Partial":
            return "🟡 Partial"
        else:
            return "🔴 No"

    if "Hands On Experience" in ai_df.columns:
        ai_df["Hands On Experience"] = ai_df["Hands On Experience"].apply(color_hands_on)

    # Clean score display for table
    for score_col in ["Industry Match Score", "Hands On Score"]:
        if score_col in ai_df.columns:
            ai_df[score_col] = ai_df[score_col].apply(
                lambda x: x.split("|")[0].strip() if isinstance(x, str) and "|" in x else x
            )

    columns_to_show = [
        "Name", "Overall Score", "Industry Match Score", "Hands On Score",
        "Current Designation", "Current Organization", "Industry",
        "Hands On Experience", "Short Description", "LinkedIn Profile"
    ]
    ai_df = ai_df[[col for col in columns_to_show if col in ai_df.columns]]
    st.write(ai_df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ------------------ DISPLAY FULL RESULTS ------------------
def display_mentor_results(ai_recommendations, df):
    if not ai_recommendations:
        return

    # Split into Tier 1 and Tier 2
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
            display_mentor_card(mentor, i + 1, "Tier 1", df)
        display_table(tier1, df, "📋 Tier 1 Summary Table")
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
            display_mentor_card(mentor, i + 1, "Tier 2", df)
        display_table(tier2, df, "📋 Tier 2 Summary Table")


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
user_input = st.chat_input("Describe your business, ask a follow-up, or start a new search...")

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
                    enriched_query = f"{user_input}\n\nContext from business document:\n{founder_doc_text[:1500]}"

                query_vec = model.encode([enriched_query])
                similarity = cosine_similarity(query_vec, vectors)
                df["score"] = similarity[0]
                candidates = df.sort_values(by="score", ascending=False).head(15)

                mentor_info = ""
                for _, row in candidates.iterrows():
                    mentor_info += f"""
Name: {row['Name']}
Expertise: {row['Expertise']}
Secondary Expertise: {row['Secondary Expertise']}
Industry: {row['Industry']}
Current Designation: {row['Current Designation']}
Current Organization: {row['Current Organization']}
Qualification: {row['Qualification']}
Description: {row['Description']}
Document Summary: {row['Doc Text'][:500] if row['Doc Text'] else 'Not available'}
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

TIER 1 — Strong Match (show minimum 1, maximum 5):
A mentor qualifies for Tier 1 ONLY if BOTH conditions are true:
  ✅ Condition 1 — Industry Match: The mentor has directly worked IN the same 
     or very closely related industry as the founder's business. Not just advised 
     — actually worked in it as an operator, founder, or senior leader.
  ✅ Condition 2 — Hands-On Experience: The mentor has PERSONALLY done the 
     specific task or solved the specific problem the founder is facing. 
     Not consulting, not teaching, not advising — actually done it themselves.

If even ONE condition is missing → mentor goes to Tier 2, NOT Tier 1.
Be strict. It is better to show 1 Tier 1 mentor than to incorrectly 
promote a weak match to Tier 1.

TIER 2 — Partial Match (maximum 5):
Mentors who meet at least ONE of the following:
  - Matches the industry but lacks hands-on experience in the specific problem
  - Has hands-on experience in the problem area but from a different industry
  - Has strong relevant expertise that could still be useful to the founder

SCORING (apply to all mentors regardless of tier):
- Industry Match: 3 points
- Hands-On Experience: 3 points
- Relevant Expertise: 2 points
- Qualification + Credibility: 2 points

STRICT RULES:
- Never put a mentor in Tier 1 just because they are impressive or well-qualified
- Industry match alone is NOT enough for Tier 1
- Hands-on alone is NOT enough for Tier 1
- BOTH must be present for Tier 1
- Be honest about Hands On Experience: Yes = personally done it, 
  Partial = advised/consulted on it, No = no relevant experience

Return strictly as a JSON array. Include ALL mentors evaluated — 
Tier 1 first (1-5 mentors), then Tier 2 (up to 5 mentors).
Total array can have between 2 and 10 objects.

Format:
[
  {{
    "Tier": "1",
    "Name": "mentor name exactly as given",
    "Overall Score": "score out of 10 as number only e.g. 8",
    "Industry Match Score": "score | one line explanation e.g. 3 | Worked in manufacturing exports for 10 years",
    "Hands On Score": "score | one line explanation e.g. 3 | Personally handled DGFT and LC documentation",
    "Expertise Score": "score | one line explanation e.g. 2 | Strong supply chain expertise",
    "Credibility Score": "score | one line explanation e.g. 1 | MBA from regional institute",
    "Match Reason": "2-3 lines — lead with WHY they qualify for this tier based on industry + hands-on",
    "Relevant Experience": "specific experience directly relevant to the founder's problem",
    "Current Designation": "their current designation",
    "Current Organization": "their current organization",
    "Qualification": "their qualification",
    "Hands On Experience": "Yes / No / Partial",
    "Hands On Details": "1-2 lines on what they have personally done. If No/Partial, state clearly what is missing."
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
                        ai_recommendations = json.loads(match_json.group()) if match_json else []

                    if ai_recommendations:
                        st.session_state.last_recommendations = ai_recommendations
                        st.session_state.last_query = user_input

                        tier1_count = len([m for m in ai_recommendations if m.get("Tier") == "1"])
                        tier2_count = len([m for m in ai_recommendations if m.get("Tier") == "2"])

                        summary = (
                            f"Found **{tier1_count} Tier 1 mentor(s)** "