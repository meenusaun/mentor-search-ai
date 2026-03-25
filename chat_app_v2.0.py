import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import anthropic
import os

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

col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.image("DP_BG1.png", width=150)

st.write("")

with col2:
    st.markdown(
        "<h2 style='text-align: center;'>🌐 Resources Network - Look for Mentor</h2>",
        unsafe_allow_html=True
    )

# ------------------ AI MODEL SELECTOR (Sidebar) ------------------
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

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl")

    df["Expertise"] = df["Expertise"].fillna("").astype(str)
    df["Secondary Expertise"] = df["Secondary Expertise"].fillna("").astype(str)
    df["Industry"] = df["Industry"].fillna("").astype(str)
    df["Secondary Industry"] = df["Secondary Industry"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Expertise Tags"] = df["Expertise Tags"].fillna("").astype(str)
    df["Industry Tags"] = df["Industry Tags"].fillna("").astype(str)

    df["combined"] = (
        "Expertise: " + df["Expertise"] + ". " +
        "Secondary Expertise: " + df["Secondary Expertise"] + ". " +
        "Industry: " + df["Industry"] + ". " +
        "Secondary Industry: " + df["Secondary Industry"] + ". " +
        "Description: " + df["Description"] + ". " +
        "Tags: " + df["Expertise Tags"] + " " + df["Industry Tags"]
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

    # -------- SEMANTIC SEARCH --------
    query_vec = model.encode([user_input])
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
"""

    prompt = f"""
User is looking for a mentor: "{user_input}"

Here are some mentors:

{mentor_info}

Task:
1. Recommend top 3 mentors
2. Explain WHY each is suitable
3. Keep response simple and structured
"""

    # -------- AI CALL (conditional on selected model) --------
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
        display_df["LinkedIn Profile"] = display_df["LinkedIn Profile"].fillna("").astype(str).apply(make_clickable)

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
            if pd.notna(row.get("LinkedIn", "")) and row.get("LinkedIn", "") != "":
                st.markdown(f"[🔗 View LinkedIn Profile]({row['LinkedIn']})")
                