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
    """Detect if user is asking a followup or starting a new search."""
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

# ------------------ DISPLAY MENTOR CARDS + TABLE ------------------
def display_mentor_results(ai_recommendations, df):
    """Display mentor recommendation cards and table."""

    if not ai_recommendations:
        return

    for i, mentor in enumerate(ai_recommendations):
        hands_on = mentor.get("Hands On Experience", "").strip()
        if hands_on == "Yes":
            badge = "🟢 Hands-On"
        elif hands_on == "Partial":
            badge = "🟡 Partial"
        else:
            badge = "🔴 No Direct Experience"

        with st.expander(
            f"#{i+1} — {mentor.get('Name', 'N/A')} | "
            f"{mentor.get('Current Designation', '')} at "
            f"{mentor.get('Current Organization', '')} | {badge}",
            expanded=(i == 0)
        ):
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

    # Table
    st.subheader("Top Matches (Recommended by AI)")
    ai_df = pd.DataFrame(ai_recommendations)

    linkedin_map = df.set_index("Name")["LinkedIn"].to_dict()
    ai_df["LinkedIn Profile"] = ai_df["Name"].map(linkedin_map).fillna("")

    def make_clickable(link):
        if pd.notna(link) and str(link).strip() != "":
            return f'<a href="{link}" target="_blank">View Profile</a>'
        return "Not Available"

    ai_df["LinkedIn Profile"] = ai_df["LinkedIn Profile"].apply(make_clickable)

    score_map = df.set_index("Name")["score"].to_dict() if "score" in df.columns else {}
    ai_df["Match Score"] = ai_df["Name"].map(score_map).fillna(0).round(2)

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

    columns_to_show = [
        "Name", "Qualification", "Current Designation", "Current Organization",
        "Industry", "Hands On Experience", "Short Description",
        "LinkedIn Profile", "Match Score"
    ]
    ai_df = ai_df[[col for col in columns_to_show if col in ai_df.columns]]
    st.write(ai_df.to_html(escape=False, index=False), unsafe_allow_html=True)


# ------------------ AI CALL HELPER ------------------
def call_ai(prompt, max_tokens=2048):
    """Call the selected AI model and return raw text."""
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

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Detect intent
    intent = detect_intent(user_input, st.session_state.last_recommendations)

    # -------- FOLLOWUP HANDLING --------
    if intent == "followup":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                # Build conversation history for context
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

Current top 5 recommended mentors:
{json.dumps(st.session_state.last_recommendations, indent=2)}

Founder's follow-up question: "{user_input}"

Instructions:
- Answer the follow-up question conversationally and helpfully
- If asked to compare mentors, compare them clearly with pros and cons
- If asked to refine search, explain what kind of mentor would be better
- If asked about a specific mentor, give detailed insights about them
- If asked for a different mentor type, suggest what to look for
- Reference mentor names, designations, qualifications and hands-on experience where relevant
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

                # Enrich query
                enriched_query = user_input
                if founder_doc_text:
                    enriched_query = f"{user_input}\n\nContext from business document:\n{founder_doc_text[:1500]}"

                # Semantic search
                query_vec = model.encode([enriched_query])
                similarity = cosine_similarity(query_vec, vectors)
                df["score"] = similarity[0]
                candidates = df.sort_values(by="score", ascending=False).head(10)

                # Build mentor info
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

                # Previous context if exists
                previous_context = ""
                if st.session_state.last_query:
                    previous_context = f"""
Note: The founder previously searched for: "{st.session_state.last_query}"
This is a new search request. Treat it independently but keep previous context in mind.
"""

                prompt = f"""
You are helping an Indian founder find the right mentor.

Founder's business brief and problem statement:
"{user_input}"

{founder_context_section}
{previous_context}

Here are mentor profiles to evaluate:
{mentor_info}

Task:
1. Identify the most likely industry based on the business brief
2. Recommend the top 5 most suitable mentors for this founder
3. For each mentor explain:
   - Why they are suitable based on their experience and background
   - Mention their Current Designation and Current Organization explicitly
   - Mention their Qualification and how it adds value
   - Which specific past experience or skill makes them relevant to the founder's problem
   - Whether the mentor has direct hands-on experience in the area the founder needs help with.
     Hands-on means they have personally done it — not just advised or consulted.
     Examples: personally managed exports, built a sales team, raised funding,
     ran a manufacturing unit, dealt with DGFT or customs, etc.
4. Return strictly as a JSON array with exactly 5 objects:

[
  {{
    "Name": "mentor name exactly as given",
    "Match Reason": "2-3 lines on why this mentor is suitable",
    "Relevant Experience": "specific experience relevant to the problem",
    "Current Designation": "their current designation",
    "Current Organization": "their current organization",
    "Qualification": "their qualification",
    "Hands On Experience": "Yes / No / Partial",
    "Hands On Details": "1-2 lines on what they have personally done relevant to founder's problem. If No, mention what is missing."
  }}
]

Return only the JSON array. No extra text, no markdown outside the array.
"""

                try:
                    ai_raw = call_ai(prompt, max_tokens=2048)

                    # Parse JSON
                    cleaned = re.sub(r"```json|```", "", ai_raw).strip()
                    try:
                        ai_recommendations = json.loads(cleaned)
                    except json.JSONDecodeError:
                        match_json = re.search(r'\[.*\]', cleaned, re.DOTALL)
                        ai_recommendations = json.loads(match_json.group()) if match_json else []

                    if ai_recommendations:
                        # Save to session state
                        st.session_state.last_recommendations = ai_recommendations
                        st.session_state.last_query = user_input

                        summary = f"Here are the top 5 mentors for your requirement. You can ask me to **compare any two**, **tell me more about a specific mentor**, **refine the search**, or **start a new search** anytime."
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
                        fallback = "I could not find strong matches. Could you describe your business problem in a bit more detail?"
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
                    founder_context_section = f"Founder uploaded a business document:\n{founder_doc_text[:1500]}"

                scoring_prompt = f"""
A founder is looking for a mentor with this requirement:
"{st.session_state.last_query}"

{founder_context_section}

AI top 5 recommended mentors:
{json.dumps(st.session_state.last_recommendations, indent=2)}

Evaluate this uploaded mentor profile:
{mentor_profile_text[:2000]}

Return strictly as a JSON object:
{{
  "Uploaded Mentor Name": "name if found, else 'Uploaded Mentor'",
  "Match Score": "score out of 10",
  "Hands On Experience": "Yes / No / Partial",
  "Hands On Details": "1-2 lines on what they have personally done relevant to founder's problem",
  "Match Summary": "2-3 lines on how well this mentor matches",
  "Key Strengths": "top 3 strengths relevant to the founder's problem",
  "Gaps": "any gaps compared to what the founder needs",
  "Rank vs Top 5": "how this mentor ranks vs AI top 5"
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
                        score_result = json.loads(match_score.group()) if match_score else {}

                    if score_result:
                        mentor_name = score_result.get("Uploaded Mentor Name", "Uploaded Mentor")
                        score_val = score_result.get("Match Score", "N/A")
                        hands_on_val = score_result.get("Hands On Experience", "").strip()

                        st.markdown(f"---\n#### 📊 Uploaded Mentor Analysis — {mentor_name}")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("🎯 Match Score", f"{score_val} / 10")
                        with col2:
                            st.markdown("**🛠️ Hands-On Experience**")
                            if hands_on_val == "Yes":
                                st.success(f"🟢 Yes — {score_result.get('Hands On Details', '')}")
                            elif hands_on_val == "Partial":
                                st.warning(f"🟡 Partial — {score_result.get('Hands On Details', '')}")
                            else:
                                st.error(f"🔴 No — {score_result.get('Hands On Details', '')}")
                        with col3:
                            st.markdown("**📊 Rank vs AI Top 5**")
                            st.write(score_result.get("Rank vs Top 5", "N/A"))

                        st.markdown("---")
                        col4, col5 = st.columns(2)
                        with col4:
                            st.markdown("**✅ Key Strengths**")
                            st.write(score_result.get("Key Strengths", "N/A"))
                        with col5:
                            st.markdown("**⚠️ Gaps**")
                            st.write(score_result.get("Gaps", "N/A"))

                        st.markdown("**📝 Match Summary**")
                        st.write(score_result.get("Match Summary", "N/A"))

                        score_content = f"Uploaded mentor **{mentor_name}** scored **{score_val}/10**. {score_result.get('Match Summary', '')}"
                        st.session_state.messages.append({
                            "role": "assistant",
                            "type": "mentor_score",
                            "content": score_content
                        })

                except Exception as e:
                    st.error(f"Mentor Scoring Error: {e}")