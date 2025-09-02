import streamlit as st
import pandas as pd
import numpy as np
import glob

# ============================
# Load dataset + similarity matrix
# ============================
# Replace with your actual files
@st.cache_data
def load_data():
    dataset = pd.read_csv("anime-transformed-dataset-2023.csv")   # your preprocessed dataset with all_titles
    return dataset

def load_cosine_similarity():
    # Load all chunk files
    parts = []
    for file in sorted(glob.glob("cosine_part_*.npz")):
        arr = np.load(file)["data"]
        parts.append(arr)

    # Stitch back into full matrix
    cosine_similarity = np.vstack(parts)
    return cosine_similarity

dataset = load_data()
cosine_similarity = load_cosine_similarity()

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
def get_recommendations(dataset, title, *, animes_indices, cosine_similarity, number_recommendations=10):
    title = title.strip().lower()
    if title not in animes_indices:
        raise KeyError(f"Title '{title}' not found in dataset (check all_titles)")
    idx = animes_indices[title]

    similarity_scores = list(enumerate(cosine_similarity[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: (x[1], dataset.iloc[int(x[0])].get("score", -1)),
        reverse=True
    )

    similarity_scores = similarity_scores[1:]  # skip itself

    similarity_scores = [
        (int(i), sim) for i, sim in similarity_scores
        if dataset.iloc[int(i)].get("score", -1) != -1
    ]

    similarity_scores = similarity_scores[:number_recommendations]

    recommended_indices = [int(pair[0]) for pair in similarity_scores]
    recommended_scores  = [pair[1] for pair in similarity_scores]

    cols = ['title', 'all_titles', 'synopsis_full', 'score', 'genres', 'themes', 'main_picture']
    cols = [c for c in cols if c in dataset.columns]
    recommendations_df = dataset.iloc[recommended_indices][cols].copy()
    recommendations_df['cosine_similarity'] = recommended_scores
    recommendations_df.insert(0, "rank", range(1, len(recommendations_df)+1))

    return recommendations_df

# ============================
# Streamlit UI
# ============================
st.title("🎥 Anime Recommendation System")
st.write("Search an anime and get top recommendations based on synopsis similarity.")

# User input
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
                cols = st.columns([1, 3])  # left for image, right for info

                # Left column: cover image if available
                with cols[0]:
                    if pd.notna(row.get("main_picture", "")) and str(row["main_picture"]).strip():
                        st.image(row["main_picture"], use_container_width=True)
                    else:
                        st.write("No image available")

                # Right column: text info
                with cols[1]:
                    st.subheader(f"{row['rank']}. {row['title']} (Score: {row['score']})")
                    st.caption(f"Genres: {row.get('genres', '-')}")
                    st.caption(f"Themes: {row.get('themes', '-')}")
                    st.write(row.get("synopsis_full", ""))
                    st.write(f"🔗 Similarity: {row['cosine_similarity']:.4f}")

                st.markdown("---")

    except KeyError as e:
        st.error(str(e))
