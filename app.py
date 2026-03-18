import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load Excel
df = pd.read_excel("mentors.xlsx")

# Combine columns into one text field
df["combined"] = df["Expertise"] + " " + df["Industry"] + " " + df["Description"]

# Convert text to vectors
model = SentenceTransformer('all-MiniLM-L6-v2')
vectors = model.encode(df["combined"])

# Streamlit UI
st.title("🔍 AI Mentor Search")

query = st.text_input("Type what you need (e.g. fundraising mentor)")

if query:
    # Clear previous results
    st.write("")

    query_vec = model.encode([query])
    
    similarity = cosine_similarity(query_vec, vectors)
    
    df["score"] = similarity[0]
    
    # Filter low scores (important fix)
    filtered_df = df[df["score"] > 0.04]

    results = filtered_df.sort_values(by="score", ascending=False).head(5)

if filtered_df.empty:
    st.warning("No strong matches found. Showing closest results.")
    results = df.sort_values(by="score", ascending=False).head(5)
else:
    results = filtered_df.sort_values(by="score", ascending=False).head(5)

    # Show fresh results
    st.subheader("Top Matches:")
    
    for index, row in results.iterrows():
        st.markdown(f"### {row['Name']}")
        st.write(f"Expertise: {row['Expertise']}")
        st.write(f"Industry: {row['Industry']}")
        st.write(f"Match Score: {round(row['score'], 2)}")

        # AI-style explanation
        st.write("Why this mentor:")
        st.write(f"- Matches expertise: {row['Expertise']}")
        st.write(f"- Relevant industry: {row['Industry']}")
        st.write("---")
        st.write(f"Industry: {row['Industry']}")
        st.write("---")