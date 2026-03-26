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

# ------------------ FILE UPLOADER 1: FOUNDER DOC ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Upload Business Document")
st.sidebar.caption("Upload your pitch deck, business plan, or any document describing your business and problem.")
founder_uploaded_file = st.sidebar.file_uploader(
    "Pitch Deck / Business Document",
    type=["pdf", "docx", "txt"],
    key="founder_doc",
    help="This helps the AI better understand your business context and find relevant mentors."
)

# ------------------ FILE UPLOADER 2: MENTOR PROFILE ------------------
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Check a Mentor's Profile")
st.sidebar.caption("Upload a mentor's resume or profile to check how well they match your requirement.")
mentor_uploaded_file = st.sidebar.file_uploader(
    "Mentor Resume / Profile",
    type=["pdf", "docx", "txt"],
    key="mentor_profile",
    help="The app will score this mentor against the AI top 5 recommendations."
)

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
        st.sidebar.success("✅ Business document parsed successfully!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from business document.")

mentor_profile_text = ""
if mentor_uploaded_file:
    mentor_profile_text = extract_text_from_uploaded_file(mentor_uploaded_file)
    if mentor_profile_text:
        st.sidebar.success("✅ Mentor profile parsed successfully!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from mentor profile.")

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("Describe your business and the problem you need mentor help with...")

# ------------------ PROCESS INPUT ------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # -------- ENRICH QUERY WITH FOUNDER DOC --------
    enriched_query = user_input
    if founder_doc_text:
        enriched_query = f"{user_input}\n\nAdditional context from business document:\n{founder_doc_text[:1500]}"

    # -------- SEMANTIC SEARCH --------
    query_vec = model.encode([enriched_query])
    similarity = cosine_similarity(query_vec, vectors)
    df["score"] = similarity[0]
    candidates = df.sort_values(by="score", ascending=False).head(10)

    # -------- BUILD MENTOR INFO FOR PROMPT --------
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
The founder also uploaded a business document with the following content:
{founder_doc_text[:1500]}
"""

    # -------- BUILD MAIN PROMPT --------
    prompt = f"""
User is looking for a mentor based on this business brief and problem statement:
"{user_input}"

{founder_context_section}

Here are mentor profiles to evaluate:
{mentor_info}

Task:
1. First identify the most likely industry based on the business brief
2. From the mentors provided, recommend the top 5 most suitable mentors for this founder
3. For each mentor explain:
   - Why they are suitable based on their experience and background
   - Mention their Current Designation and Current Organization explicitly
   - Mention their Qualification and how it adds value
   - Which specific past experience or skill makes them relevant to the founder's problem
4. Keep response simple, structured and founder-friendly
5. Return your answer strictly as a JSON array with exactly 5 objects in this format:

[
  {{
    "Name": "mentor name exactly as given",
    "Match Reason": "2-3 lines on why this mentor is suitable for the founder's specific problem",
    "Relevant Experience": "specific experience or skill relevant to the problem",
    "Current Designation": "their current designation",
    "Current Organization": "their current organization",
    "Qualification": "their qualification"
  }}
]

Return only the JSON array. No extra text, no markdown, no explanation outside the array.
"""

    # -------- AI CALL --------
    try:
        st.markdown(f"### 🤖 AI Recommendation — {ai_model}")

        if ai_model == "GPT-4o Mini (OpenAI)":
            response_ai = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_raw = response_ai.choices[0].message.content
        else:
            response_ai = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_raw = response_ai.content[0].text

        # -------- PARSE JSON --------
        cleaned = re.sub(r"```json|```", "", ai_raw).strip()
        try:
            ai_recommendations = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            ai_recommendations = json.loads(match.group()) if match else []

        # -------- SHOW AI RECOMMENDATION CARDS --------
        if ai_recommendations:
            for i, mentor in enumerate(ai_recommendations):
                with st.expander(
                    f"#{i+1} — {mentor.get('Name', 'N/A')} | "
                    f"{mentor.get('Current Designation', '')} at "
                    f"{mentor.get('Current Organization', '')}",
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

                    matched_row = df[df["Name"] == mentor.get("Name")]
                    if not matched_row.empty:
                        linkedin = matched_row.iloc[0].get("LinkedIn", "")
                        if pd.notna(linkedin) and str(linkedin).strip() != "":
                            st.markdown(f"[🔗 View LinkedIn Profile]({linkedin})")
        else:
            st.warning("Could not parse AI recommendations. Showing raw response.")
            with st.chat_message("assistant"):
                st.write(ai_raw)

        st.session_state.messages.append({"role": "assistant", "content": ai_raw})

        # -------- TABLE FROM AI RESULTS --------
        st.subheader("Top Matches (Recommended by AI)")

        if ai_recommendations:
            ai_df = pd.DataFrame(ai_recommendations)

            linkedin_map = df.set_index("Name")["LinkedIn"].to_dict()
            ai_df["LinkedIn Profile"] = ai_df["Name"].map(linkedin_map).fillna("")

            def make_clickable(link):
                if pd.notna(link) and str(link).strip() != "":
                    return f'<a href="{link}" target="_blank">View Profile</a>'
                return "Not Available"

            ai_df["LinkedIn Profile"] = ai_df["LinkedIn Profile"].apply(make_clickable)

            score_map = df.set_index("Name")["score"].to_dict()
            ai_df["Match Score"] = ai_df["Name"].map(score_map).fillna(0).round(2)

            desc_map = df.set_index("Name")["Description"].to_dict()
            ai_df["Short Description"] = ai_df["Name"].map(desc_map).apply(
                lambda x: (x[:100] + "...") if isinstance(x, str) and len(x) > 100 else x
            )

            industry_map = df.set_index("Name")["Industry"].to_dict()
            ai_df["Industry"] = ai_df["Name"].map(industry_map).fillna("")

            columns_to_show = [
                "Name", "Qualification", "Current Designation", "Current Organization",
                "Industry", "Short Description", "LinkedIn Profile", "Match Score"
            ]
            ai_df = ai_df[[col for col in columns_to_show if col in ai_df.columns]]
            st.write(ai_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        # -------- UPLOADED MENTOR PROFILE SCORING --------
        if mentor_profile_text and ai_recommendations:
            st.markdown("---")
            st.subheader("📊 Uploaded Mentor Profile — Match Analysis")

            scoring_prompt = f"""
A founder is looking for a mentor with this requirement:
"{user_input}"

{founder_context_section}

The AI has already recommended these top 5 mentors:
{json.dumps(ai_recommendations, indent=2)}

Now evaluate this uploaded mentor profile against the founder's requirement 
and compare it with the top 5 AI recommended mentors:

Uploaded Mentor Profile:
{mentor_profile_text[:2000]}

Return strictly as a JSON object in this format:
{{
  "Uploaded Mentor Name": "name if found in profile, else 'Uploaded Mentor'",
  "Match Score": "score out of 10",
  "Match Summary": "2-3 lines on how well this mentor matches the founder's requirement",
  "Key Strengths": "top 3 strengths relevant to the founder's problem",
  "Gaps": "any gaps compared to what the founder needs",
  "Rank vs Top 5": "how this mentor ranks compared to AI top 5 (e.g. Better than #3 and #4, weaker than #1 and #2)"
}}

Return only the JSON object. No extra text.
"""

            try:
                if ai_model == "GPT-4o Mini (OpenAI)":
                    score_response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": scoring_prompt}]
                    )
                    score_raw = score_response.choices[0].message.content
                else:
                    score_response = anthropic_client.messages.create(
                        model="claude-sonnet-4-5",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": scoring_prompt}]
                    )
                    score_raw = score_response.content[0].text

                score_cleaned = re.sub(r"```json|```", "", score_raw).strip()
                try:
                    score_result = json.loads(score_cleaned)
                except json.JSONDecodeError:
                    match = re.search(r'\{.*\}', score_cleaned, re.DOTALL)
                    score_result = json.loads(match.group()) if match else {}

                if score_result:
                    # Score card display
                    score_val = score_result.get("Match Score", "N/A")
                    mentor_name = score_result.get("Uploaded Mentor Name", "Uploaded Mentor")

                    st.markdown(f"#### 👤 {mentor_name}")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🎯 Match Score", f"{score_val} / 10")
                    with col2:
                        st.markdown("**📊 Rank vs AI Top 5**")
                        st.write(score_result.get("Rank vs Top 5", "N/A"))
                    with col3:
                        st.markdown("**📝 Summary**")
                        st.write(score_result.get("Match Summary", "N/A"))

                    st.markdown("---")
                    col4, col5 = st.columns(2)
                    with col4:
                        st.markdown("**✅ Key Strengths**")
                        st.write(score_result.get("Key Strengths", "N/A"))
                    with col5:
                        st.markdown("**⚠️ Gaps**")
                        st.write(score_result.get("Gaps", "N/A"))
                else:
                    st.warning("Could not parse mentor profile score.")
                    st.write(score_raw)

            except Exception as e:
                st.error(f"Mentor Profile Scoring Error: {e}")

        elif mentor_profile_text and not ai_recommendations:
            st.info("💡 Mentor profile uploaded. Run a search first to get match analysis.")

    except Exception as e:
        st.error(f"AI Error: {e}")