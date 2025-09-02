import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel   
import string                                              

# ============================
# Load dataset
# ============================
@st.cache_data
def load_data():
    return pd.read_csv("anime-transformed-dataset-2023.csv")   # your dataset

# ============================
# Load cosine similarity matrix
# ============================
dataset = load_data()


dataset = dataset[dataset["transformed_synopsis_full"].notna()]
dataset["transformed_synopsis_full"] = dataset["transformed_synopsis_full"].astype(str)

tfidf_vectorizer = TfidfVectorizer(analyzer='word', norm='l2', stop_words='english')
dataset_tfidf_synopsis_full = tfidf_vectorizer.fit_transform(dataset.transformed_synopsis_full)

batch_size = 1000
n = dataset_tfidf_synopsis_full.shape[0]
cosine_similarity = np.zeros((n, n), dtype=np.float32)

for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    cosine_similarity[start:end] = linear_kernel(
        dataset_tfidf_synopsis_full[start:end],
        dataset_tfidf_synopsis_full
    )

# ============================
# Build title lookup index
# ============================
def build_title_index(dataset):
    mapping = {}
    for i, titles in enumerate(dataset["all_titles"]):
        if pd.isna(titles):
            continue
        for t in str(titles).split(","):
            t_clean = t.strip().lower()
            if t_clean and t_clean not in mapping:
                mapping[t_clean] = i
    return mapping

animes_indices = build_title_index(dataset)

# ============================
# Recommendation function
# ============================
def get_recommendations(dataset, title, *, animes_indices, tfidf_matrix, number_recommendations=10):
    title = title.strip().lower()
    if title not in animes_indices:
        raise KeyError(f"Title '{title}' not found in dataset (check all_titles)")

    idx = animes_indices[title]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-number_recommendations-2:-1]
    related_docs_indices = [i for i in related_docs_indices if i != idx]

    recommendations_df = dataset.iloc[related_docs_indices][["title","score","genres","themes","synopsis_full","main_picture"]].copy()
    recommendations_df.insert(0, "rank", range(1, len(recommendations_df)+1))
    recommendations_df["cosine_similarity"] = cosine_similarities[related_docs_indices]
    return recommendations_df

# ============================
# Streamlit UI
# ============================
st.title("🎥 Anime Recommendation System")
st.write("Search an anime and get top recommendations based on synopsis similarity.")

search_title = st.text_input("Enter an anime title (English, Japanese, or synonym):")

if search_title:
    try:
        recs = get_recommendations(
            dataset,
            search_title,
            animes_indices=animes_indices,
            cosine_similarity=cosine_similarity,
            number_recommendations=10
        )

        st.success(f"Recommendations for **{search_title}**:")
        for _, row in recs.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    if pd.notna(row.get("main_picture", "")) and str(row["main_picture"]).strip():
                        st.image(row["main_picture"], use_container_width=True)
                    else:
                        st.write("No image available")
                with cols[1]:
                    st.subheader(f"{row['rank']}. {row['title']} (Score: {row['score']})")
                    st.caption(f"Genres: {row.get('genres', '-')}")
                    st.caption(f"Themes: {row.get('themes', '-')}")
                    st.write(row.get("synopsis_full", ""))
                    st.write(f"🔗 Similarity: {row['cosine_similarity']:.4f}")
            st.markdown("---")
    except KeyError as e:
        st.error(str(e))
