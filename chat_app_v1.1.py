import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Resources Network - Look for Mentor",
    page_icon="🔍",
    layout="wide"
)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.image("DP_BG1.png", width=150)

st.write("")  # adds spacing

with col2:
    st.markdown(
        "<h2 style='text-align: center;'>🌐 Resources Network - Look for Mentor</h2>",
        unsafe_allow_html=True
    )

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl")
    # Fill missing values safely
    df["Expertise"] = df["Expertise"].fillna("").astype(str)
    df["Industry"] = df["Industry"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)

    # Combine text
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
    display_df["row_id"] = display_df.index

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
        display_df["LinkedIn Profile"] = display_df["LinkedIn Profile"].fillna("").astype(str).apply(make_clickable)

    # Select columns
    columns_to_show = ["Name", "Expertise", "Industry", "Short Description", "LinkedIn Profile", "Match Score"]
    display_df = display_df[[col for col in columns_to_show if col in display_df.columns]]

    # Show table
    st.subheader("Top Matches")

    for idx, row in results.iterrows():
        st.markdown(f"### {row['Name']}")
    st.write(f"**Expertise:** {row['Expertise']}")
    st.write(f"**Industry:** {row['Industry']}")

    short_desc = row["Description"][:100] + "..." if isinstance(row["Description"], str) else ""
    st.write(f"**Description:** {short_desc}")

    # View Details button
    if st.button(f"View Details - {idx}"):
        full_desc = row["Description"]
        if isinstance(full_desc, str) and full_desc.strip() != "":
            st.info(full_desc)
        else:
            st.warning("Description not available")

    # LinkedIn
    if pd.notna(row.get("LinkedIn", "")):
        st.markdown(f"[🔗 View LinkedIn Profile]({row['LinkedIn']})")

    st.write("---")