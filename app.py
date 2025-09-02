import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import linear_kernel

# ============================
# Load dataset
# ============================
@st.cache_data
def load_data():
    df = pd.read_csv("anime-transformed-dataset-2023.csv")

    # Drop missing synopses
    df = df[df["transformed_synopsis_full"].notna()]
    df["transformed_synopsis_full"] = df["transformed_synopsis_full"].astype(str)

    # Build all_titles if missing
    if "all_titles" not in df.columns:
        title_cols = [c for c in ["title", "title_english", "title_japanese", "title_synonyms"] if c in df.columns]
        df["all_titles"] = df[title_cols].fillna("").agg(",".join, axis=1)

    return df

dataset = load_data()

# ============================
# TF-IDF (fit once)
# ============================
@st.cache_resource
def build_tfidf_matrix(texts):
    tfidf = TfidfVectorizer(analyzer='word', norm='l2', stop_words='english')
    matrix = tfidf.fit_transform(texts)
    return tfidf, matrix

tfidf_vectorizer, dataset_tfidf_synopsis_full = build_tfidf_matrix(dataset.transformed_synopsis_full)

# ============================
# Title index
# ============================
def build_title_index(dataset):
    mapping = {}
    for i, titles in enumerate(dataset["all_titles"]):
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

    # Compute similarity on demand
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    related_indices = cosine_similarities.argsort()[::-1]  # descending order

    # Exclude itself
    related_indices = [i for i in related_indices if i != idx]

    # Keep top N
    top_indices = related_indices[:number_recommendations]
    top_scores = cosine_similarities[top_indices]

    # Columns to show
    cols = ['title', 'all_titles', 'synopsis_full', 'score', 'genres', 'themes', 'main_picture']
    cols = [c for c in cols if c in dataset.columns]

    recommendations_df = dataset.iloc[top_indices][cols].copy()
    recommendations_df['cosine_similarity'] = top_scores
    recommendations_df.insert(0, "rank", range(1, len(recommendations_df)+1))
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
            tfidf_matrix=dataset_tfidf_synopsis_full,
            number_recommendations=10
        )

        st.success(f"Recommendations for **{search_title}**:")
        for _, row in recs.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                with cols[0]:
                    if pd.notna(row.get("main_picture", "")) and str(row["main_picture"]).strip():
                        try:
                            st.image(row["main_picture"], use_container_width=True)
                        except:
                            st.write("Image not available")
                    else:
                        st.write("No image available")
                with cols[1]:
                    st.subheader(f"{row['rank']}. {row['title']} (Score: {row.get('score', '-')})")
                    st.caption(f"Genres: {row.get('genres', '-')}")
                    st.caption(f"Themes: {row.get('themes', '-')}")
                    st.write(row.get("synopsis_full", ""))
                    st.write(f"🔗 Similarity: {row['cosine_similarity']:.4f}")
            st.markdown("---")
    except KeyError as e:
        st.error(str(e))
