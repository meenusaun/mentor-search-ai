import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 AI Mentor Search")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl")
    df["combined"] = df["Expertise"] + " " + df["Industry"] + " " + df["Description"]
    return df

df = load_data()

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')

model = load_model()

# ------------------ CREATE VECTORS ------------------
@st.cache_data
def get_vectors(texts):
    return model.encode(texts)

vectors = get_vectors(df["combined"])

# ------------------ CHAT HISTORY ------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ------------------ USER INPUT ------------------
user_input = st.chat_input("Ask me to find a mentor...")

# ------------------ PROCESS INPUT ------------------
if user_input:

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # -------- RESULT LOGIC --------
    query_vec = model.encode([user_input])
    similarity = cosine_similarity(query_vec, vectors)
    df["score"] = similarity[0]

    filtered_df = df[df["score"] > 0.4]

    if filtered_df.empty:
        st.warning("No strong matches found. Showing closest results.")
        results = df.sort_values(by="score", ascending=False).head(5)
    else:
        results = filtered_df.sort_values(by="score", ascending=False).head(5)

    # -------- CHAT RESPONSE --------
    response = "Here are the best mentors for you:\n\n"

    for _, row in results.iterrows():
        response += f"👉 {row['Name']} ({row['Expertise']})\n"

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

    # -------- TABLE DISPLAY --------
    st.subheader("Top Matches")

    display_df = results.copy()

    # Rename columns
    display_df.rename(columns={
        "score": "Match Score",
        "LinkedIn": "LinkedIn Profile"
    }, inplace=True)

    # Round score
    display_df["Match Score"] = display_df["Match Score"].round(2)

    # Short description
    def shorten_text(text, length=100):
        if isinstance(text, str) and len(text) > length:
            return text[:length] + "..."
        return text

    display_df["Short Description"] = display_df["Description"].apply(shorten_text)

    # Clickable LinkedIn
    def make_clickable(link):
        if pd.notna(link) and link != "":
            return f'<a href="{link}" target="_blank">View Profile</a>'
        return "Not Available"

    if "LinkedIn Profile" in display_df.columns:
        display_df["LinkedIn Profile"] = display_df["LinkedIn Profile"].apply(make_clickable)

    # Select columns
    columns_to_show = ["Name", "Expertise", "Industry", "Short Description", "LinkedIn Profile", "Match Score"]
    display_df = display_df[[col for col in columns_to_show if col in display_df.columns]]

    # Show table
    st.write(
        display_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    # -------- FULL DESCRIPTION --------
    st.subheader("View Full Description")

    selected_name = st.selectbox(
        "Select a mentor",
        display_df["Name"]
    )

    if selected_name:
        full_desc = df[df["Name"] == selected_name]["Description"].values[0]
        st.info(full_desc)