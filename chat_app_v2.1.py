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

st.sidebar.markdown("---")
st.sidebar.subheader("📄 Upload Your Profile (Optional)")
user_uploaded_file = st.sidebar.file_uploader(
    "Upload your resume or requirements doc",
    type=["pdf", "docx", "txt"],
    help="Upload a PDF, Word doc, or text file to help find better mentor matches."
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

    if "Document Path" not in df.columns:
        df["Document Path"] = ""
    if "LinkedIn" not in df.columns:
        df["LinkedIn"] = ""
    if "Qualification" not in df.columns:
        df["Qualification"] = ""
    if "Current Organization" not in df.columns:
        df["Current Organization"] = ""
    if "Current Designation" not in df.columns:
        df["Current Designation"] = ""

    df["Document Path"] = df["Document Path"].fillna("").astype(str)
    df["LinkedIn"] = df["LinkedIn"].fillna("").astype(str)
    df["Qualification"] = df["Qualification"].fillna("").astype(str)
    df["Current Organization"] = df["Current Organization"].fillna("").astype(str)
    df["Current Designation"] = df["Current Designation"].fillna("").astype(str)

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

# ------------------ EXTRACT USER UPLOAD TEXT ------------------
user_doc_text = ""
if user_uploaded_file:
    user_doc_text = extract_text_from_uploaded_file(user_uploaded_file)
    if user_doc_text:
        st.sidebar.success("✅ Document parsed successfully!")
    else:
        st.sidebar.warning("⚠️ Could not extract text from the file.")

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

    # -------- ENRICH QUERY WITH USER DOC --------
    enriched_query = user_input
    if user_doc_text:
        enriched_query = f"{user_input}\n\nAdditional context from uploaded document:\n{user_doc_text[:1500]}"

    # -------- SEMANTIC SEARCH (to get candidate pool) --------
    query_vec = model.encode([enriched_query])
    similarity = cosine_similarity(query_vec, vectors)
    df["score"] = similarity[0]

    # Get top 10 candidates from embedding model to pass to AI
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

    user_context_section = ""
    if user_doc_text:
        user_context_section = f"""
The user also uploaded a document with the following content:
{user_doc_text[:1500]}
"""

    # -------- BUILD PROMPT --------
    prompt = f"""
User is looking for a mentor based on this business brief and problem statement:
"{user_input}"

{user_context_section}

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

        else:  # Claude (Anthropic)
            response_ai = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_raw = response_ai.content[0].text

        # -------- PARSE JSON FROM AI RESPONSE --------
        cleaned = re.sub(r"```json|```", "", ai_raw).strip()

        try:
            ai_recommendations = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r'\[.*\]', cleaned, re.DOTALL)
            if match:
                ai_recommendations = json.loads(match.group())
            else:
                ai_recommendations = []

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

                    # Match LinkedIn from original df
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

            # Merge LinkedIn from original df
            linkedin_map = df.set_index("Name")["LinkedIn"].to_dict()
            ai_df["LinkedIn Profile"] = ai_df["Name"].map(linkedin_map).fillna("")

            def make_clickable(link):
                if pd.notna(link) and str(link).strip() != "":
                    return f'<a href="{link}" target="_blank">View Profile</a>'
                return "Not Available"

            ai_df["LinkedIn Profile"] = ai_df["LinkedIn Profile"].apply(make_clickable)

            # Match Score from embedding model
            score_map = df.set_index("Name")["score"].to_dict()
            ai_df["Match Score"] = ai_df["Name"].map(score_map).fillna(0).round(2)

            # Short description from original df
            desc_map = df.set_index("Name")["Description"].to_dict()
            ai_df["Short Description"] = ai_df["Name"].map(desc_map).apply(
                lambda x: (x[:100] + "...") if isinstance(x, str) and len(x) > 100 else x
            )

            # Industry from original df
            industry_map = df.set_index("Name")["Industry"].to_dict()
            ai_df["Industry"] = ai_df["Name"].map(industry_map).fillna("")

            columns_to_show = [
                "Name", "Qualification", "Current Designation", "Current Organization",
                "Industry", "Short Description", "LinkedIn Profile", "Match Score"
            ]
            ai_df = ai_df[[col for col in columns_to_show if col in ai_df.columns]]

            st.write(ai_df.to_html(escape=False, index=False), unsafe_allow_html=True)

        else:
            # Fallback to embedding results
            st.warning("Showing embedding-based results as fallback.")
            fallback_df = candidates.head(5).copy()
            fallback_df.rename(columns={"score": "Match Score", "LinkedIn": "LinkedIn Profile"}, inplace=True)
            fallback_df["Match Score"] = fallback_df["Match Score"].round(2)
            fallback_df["Short Description"] = fallback_df["Description"].apply(
                lambda x: (x[:100] + "...") if isinstance(x, str) and len(x) > 100 else x
            )
            fallback_df["LinkedIn Profile"] = fallback_df["LinkedIn Profile"].fillna("").astype(str).apply(
                lambda link: f'<a href="{link}" target="_blank">View Profile</a>' if link.strip() != "" else "Not Available"
            )
            columns_to_show = [
                "Name", "Qualification", "Current Designation", "Current Organization",
                "Industry", "Short Description", "LinkedIn Profile", "Match Score"
            ]
            fallback_df = fallback_df[[col for col in columns_to_show if col in fallback_df.columns]]
            st.write(fallback_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"AI Error: {e}")