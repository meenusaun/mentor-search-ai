import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
import os
import pdfplumber
import docx
import requests
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
    help="Upload a PDF, Word doc, or text file. This will help find better mentor matches for you."
)

# ------------------ DOCUMENT EXTRACTION UTILS ------------------
def extract_text_from_pdf_bytes(file_bytes):
    """Extract text from PDF bytes using pdfplumber."""
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
    """Extract text from DOCX bytes."""
    text = ""
    try:
        doc = docx.Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        text = f"[DOCX extraction error: {e}]"
    return text.strip()

def extract_text_from_uploaded_file(uploaded_file):
    """Extract text from a Streamlit uploaded file."""
    if uploaded_file is None:
        return ""
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)  # reset pointer
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
    """Extract text from a local file path (for mentor docs in Excel)."""
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

    # Optional columns for doc path and LinkedIn
    if "Document Path" not in df.columns:
        df["Document Path"] = ""
    if "LinkedIn" not in df.columns:
        df["LinkedIn"] = ""

    df["Document Path"] = df["Document Path"].fillna("").astype(str)
    df["LinkedIn"] = df["LinkedIn"].fillna("").astype(str)

    # Extract mentor document text
    df["Doc Text"] = df["Document Path"].apply(extract_text_from_file_path)

    # Build combined text for semantic search
    df["combined"] = (
        "Expertise: " + df["Expertise"] + ". " +
        "Secondary Expertise: " + df["Secondary Expertise"] + ". " +
        "Industry: " + df["Industry"] + ". " +
        "Secondary Industry: " + df["Secondary Industry"] + ". " +
        "Description: " + df["Description"] + ". " +
        "Tags: " + df["Expertise Tags"] + " " + df["Industry Tags"] + ". " +
        "Document: " + df["Doc Text"].str[:1000]  # cap to avoid token overload
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
user_input = st.chat_input("Ask me to find a mentor...")

# ------------------ PROCESS INPUT ------------------
if user_input:

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # -------- ENRICH QUERY WITH USER DOC --------
    enriched_query = user_input
    if user_doc_text:
        enriched_query = f"{user_input}\n\nAdditional context from uploaded document:\n{user_doc_text[:1500]}"

    # -------- SEMANTIC SEARCH --------
    query_vec = model.encode([enriched_query])
    similarity = cosine_similarity(query_vec, vectors)
    df["score"] = similarity[0]

    filtered_df = df[df["score"] > 0.4]
    if filtered_df.empty:
        st.warning("No strong matches found. Showing closest results.")
        results = df.sort_values(by="score", ascending=False).head(5)
    else:
        results = filtered_df.sort_values(by="score", ascending=False).head(5)

    # -------- BUILD PROMPT --------
    mentor_info = ""
    for _, row in results.iterrows():
        mentor_info += f"""
Name: {row['Name']}
Expertise: {row['Expertise']}
Industry: {row['Industry']}
Description: {row['Description']}
Document Summary: {row['Doc Text'][:500] if row['Doc Text'] else 'Not available'}
"""

    user_context_section = ""
    if user_doc_text:
        user_context_section = f"""
The user also uploaded a document with the following content (use this to better understand their needs):
{user_doc_text[:1500]}
"""

    prompt = f"""
User is looking for a mentor: "{user_input}"
{user_context_section}
Here are some mentors:
{mentor_info}

Task:
1. Recommend top 3 mentors
2. Explain WHY each is suitable based on the user's query and uploaded document (if any)
3. Highlight relevant skills, experience, and background from their profiles
4. Keep response simple and structured
"""

    # -------- AI CALL --------
    try:
        st.markdown(f"### 🤖 AI Recommendation — {ai_model}")

        if ai_model == "GPT-4o Mini (OpenAI)":
            response_ai = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response_ai.choices[0].message.content

        else:  # Claude (Anthropic)
            response_ai = anthropic_client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            ai_output = response_ai.content[0].text

        with st.chat_message("assistant"):
            st.write(ai_output)

        st.session_state.messages.append({"role": "assistant", "content": ai_output})

    except Exception as e:
        st.error(f"AI Error: {e}")

    # ------------------ TABLE DISPLAY ------------------
    st.subheader("Top Matches")

    display_df = results.copy()
    display_df.rename(columns={"score": "Match Score", "LinkedIn": "LinkedIn Profile"}, inplace=True)
    display_df["Match Score"] = display_df["Match Score"].round(2)

    def shorten_text(text, length=100):
        if isinstance(text, str) and len(text) > length:
            return text[:length] + "..."
        return text

    display_df["Short Description"] = display_df["Description"].apply(shorten_text)

    def make_clickable(link):
        if pd.notna(link) and link != "":
            return f'<a href="{link}" target="_blank">View Profile</a>'
        return "Not Available"

    if "LinkedIn Profile" in display_df.columns:
        display_df["LinkedIn Profile"] = (
            display_df["LinkedIn Profile"].fillna("").astype(str).apply(make_clickable)
        )

    columns_to_show = ["Name", "Expertise", "Industry", "Short Description", "LinkedIn Profile", "Match Score"]
    display_df = display_df[[col for col in columns_to_show if col in display_df.columns]]

    st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ------------------ VIEW DETAILS ------------------
    st.subheader("View Mentor Details")

    for idx, row in results.iterrows():
        with st.expander(f"{row['Name']} ({row['Expertise']})"):
            st.write(f"**Industry:** {row['Industry']}")

            if row["Description"]:
                st.write(row["Description"])
            else:
                st.warning("Description not available")

            if row.get("Doc Text", ""):
                with st.expander("📄 Extracted from Mentor Document"):
                    st.write(row["Doc Text"][:1000] + ("..." if len(row["Doc Text"]) > 1000 else ""))

            if pd.notna(row.get("LinkedIn", "")) and row.get("LinkedIn", "") != "":
                st.markdown(f"[🔗 View LinkedIn Profile]({row['LinkedIn']})")