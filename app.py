import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🔍 AI Mentor Search")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("mentors.xlsx", engine="openpyxl")
    df["combined"] = df["Expertise"] + " " + df["Industry"] + " " + df["Description"]
    return df

df = load_data()

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L3-v2')  # lighter model

model = load_model()

# Create embeddings
@st.cache_data
def get_vectors(texts):
    return model.encode(texts)

vectors = get_vectors(df["combined"])

# User input
query = st.text_input("Type what you need (e.g. fundraising mentor)")

if query:
    query_vec = model.encode([query])
    similarity = cosine_similarity(query_vec, vectors)
    df["score"] = similarity[0]

    # Filter results
    filtered_df = df[df["score"] > 0.4]

    if filtered_df.empty:
        st.warning("No strong matches found. Showing closest results.")
        results = df.sort_values(by="score", ascending=False).head(5)
    else:
        results = filtered_df.sort_values(by="score", ascending=False).head(5)

    st.subheader("Top Matches:")

    # Create clean table
    display_df = results.copy()
    display_df.rename(columns={"score": "Match Score"}, inplace=True)

    def shorten_text(text, length=100):
        if isinstance(text, str) and len(text) > length:
            return text[:length] + "..."
        return text

    display_df["Short Description"] = display_df["Description"].apply(shorten_text)

    # Round score
    display_df["Match Score"] = display_df["Match Score"].round(2)

    display_df.rename(columns={
    "score": "Match Score",
    "LinkedIn": "LinkedIn Profile",
    "Short Description": "Description"
    }, inplace=True)

    # Convert LinkedIn to clickable links
    def make_clickable(link):
        if pd.notna(link) and link != "":
            return f'<a href="{link}" target="_blank">View Profile</a>'
        return "Not Available"

    if "LinkedIn Profile" in display_df.columns:
        display_df["LinkedIn Profile"] = display_df["LinkedIn Profile"].apply(make_clickable)

    columns_to_show = ["Name", "Expertise", "Industry", "Short Description", "LinkedIn Profile", "Match Score"]
    display_df = display_df[[col for col in columns_to_show if col in display_df.columns]]
    
    st.subheader("Top Matches:")

    st.write(
        display_df.to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

    st.subheader("Top Matches:")