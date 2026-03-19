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

    for _, row in results.iterrows():
        st.markdown(f"### {row['Name']}")
        st.write(f"Expertise: {row['Expertise']}")
        st.write(f"Industry: {row['Industry']}")
        st.write(f"Match Score: {round(row['score'], 2)}")
        st.write("Why this mentor:")
        st.write(f"- Matches expertise: {row['Expertise']}")
        st.write(f"- Relevant industry: {row['Industry']}")
        st.write("---")