
from pathlib import Path
import html

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CONTENT_WEIGHT = 0.30
CF_WEIGHT = 0.70
TOP_NEIGHBORS = 20


st.set_page_config(
    page_title="MovieLens - AI Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #10131c, #191d2b);
        color: #f5f5f5;
    }
    [data-testid="stSidebar"] {
        background-color: #151925;
    }
    .hero {
        padding: 2rem 1rem;
        border-radius: 18px;
        text-align: center;
        background: linear-gradient(120deg, #243b55, #141e30);
        border: 1px solid #343d52;
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: white;
        font-size: 2.4rem;
    }
    .hero p {
        color: #c4ccda;
    }
    .movie-card {
        background: #1c2231;
        border: 1px solid #343c50;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 155px;
    }
    .movie-title {
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }
    .movie-genres {
        color: #aebbd0;
        font-size: 0.9rem;
        margin-bottom: 0.7rem;
    }
    .movie-rating {
        color: #ffd166;
        font-weight: 600;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 9px;
        background: #4361ee;
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_data():
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    tags = pd.read_csv(DATA_DIR / "tags.csv")

    ratings["userId"] = pd.to_numeric(
        ratings["userId"], errors="coerce"
    )
    ratings["movieId"] = pd.to_numeric(
        ratings["movieId"], errors="coerce"
    )
    ratings["rating"] = pd.to_numeric(
        ratings["rating"], errors="coerce"
    )

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    ).copy()

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    tags_by_movie = (
        tags.groupby("movieId")["tag"]
        .apply(lambda values: " ".join(values.dropna().astype(str)))
        .reset_index()
    )

    movies = movies.merge(
        tags_by_movie, on="movieId", how="left"
    )

    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].fillna("")

    movies["content"] = (
        movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["tag"]
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b\w+\b",
        dtype=np.float32
    )

    tfidf_matrix = vectorizer.fit_transform(movies["content"])

    user_movie = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    )

    rating_array = user_movie.to_numpy(dtype=np.float32)
    rating_mask = np.isfinite(rating_array)

    user_means = np.nanmean(rating_array, axis=1)
    user_means = np.nan_to_num(
        user_means, nan=float(ratings["rating"].mean())
    )

    centered = np.where(
        rating_mask,
        rating_array - user_means[:, None],
        0.0
    )

    user_similarity = cosine_similarity(centered).astype(np.float32)
    np.fill_diagonal(user_similarity, 0.0)

    movie_ids = movies["movieId"].to_numpy(dtype=int)
    user_ids = user_movie.index.to_numpy(dtype=int)
    rating_movie_ids = user_movie.columns.to_numpy(dtype=int)

    movie_to_index = {
        int(movie_id): index
        for index, movie_id in enumerate(movie_ids)
    }

    user_to_index = {
        int(user_id): index
        for index, user_id in enumerate(user_ids)
    }

    movie_to_rating_col = {
        int(movie_id): index
        for index, movie_id in enumerate(rating_movie_ids)
    }

    return {
        "ratings": ratings,
        "movies": movies,
        "user_movie": user_movie,
        "rating_array": rating_array,
        "rating_mask": rating_mask,
        "user_means": user_means,
        "user_similarity": user_similarity,
        "movie_ids": movie_ids,
        "user_ids": user_ids,
        "rating_movie_ids": rating_movie_ids,
        "movie_to_index": movie_to_index,
        "user_to_index": user_to_index,
        "movie_to_rating_col": movie_to_rating_col,
        "tfidf_matrix": tfidf_matrix
    }


def recommend(data, user_id, favorite_ids, limit=10):
    user_id = int(user_id)

    if user_id not in data["user_to_index"]:
        raise ValueError(f"Unknown MovieLens user ID: {user_id}")

    user_index = data["user_to_index"][user_id]
    user_movie = data["user_movie"]
    user_ratings = user_movie.loc[user_id].dropna()

    seen_movies = set(int(mid) for mid in user_ratings.index)

    valid_favorites = [
        int(mid)
        for mid in favorite_ids
        if int(mid) in data["movie_to_index"]
    ]

    if not valid_favorites:
        valid_favorites = [
            int(mid)
            for mid, rating in user_ratings.items()
            if rating >= 4.0 and int(mid) in data["movie_to_index"]
        ]

    if not valid_favorites:
        raise ValueError(
            "Choose at least one favourite movie, or select a user "
            "with movies rated 4 stars or higher."
        )

    liked_indices = [
        data["movie_to_index"][mid] for mid in valid_favorites
    ]

    tfidf_matrix = data["tfidf_matrix"]

    user_profile = tfidf_matrix[liked_indices].mean(axis=0)

    content_scores = cosine_similarity(
        np.asarray(user_profile), tfidf_matrix
    ).ravel()

    similarities = data["user_similarity"][user_index]

    neighbor_indices = np.argsort(similarities)[::-1][
        :TOP_NEIGHBORS
    ]

    neighbor_indices = [
        idx for idx in neighbor_indices
        if similarities[idx] > 0
    ]

    rating_movie_ids = data["rating_movie_ids"]
    rating_array = data["rating_array"]
    rating_mask = data["rating_mask"]
    user_means = data["user_means"]

    cf_predictions = np.full(
        len(rating_movie_ids), np.nan, dtype=np.float32
    )

    if neighbor_indices:
        neighbor_weights = similarities[neighbor_indices]
        neighbor_ratings = rating_array[neighbor_indices]
        neighbor_masks = rating_mask[neighbor_indices]

        deviations = np.where(
            neighbor_masks,
            neighbor_ratings - user_means[neighbor_indices, None],
            0.0
        )

        weighted_sum = (
            neighbor_weights[:, None] * deviations
        ).sum(axis=0)

        weight_sum = (
            neighbor_weights[:, None] * neighbor_masks
        ).sum(axis=0)

        valid_columns = weight_sum > 0

        cf_predictions[valid_columns] = (
            user_means[user_index]
            + weighted_sum[valid_columns] / weight_sum[valid_columns]
        )

    overall_mean = float(data["ratings"]["rating"].mean())

    candidates = [
        int(mid)
        for mid in data["movie_ids"]
        if int(mid) not in seen_movies
        and int(mid) not in valid_favorites
    ]

    if not candidates:
        return [], []

    candidate_indices = np.array([
        data["movie_to_index"][mid] for mid in candidates
    ])

    candidate_content = content_scores[candidate_indices]

    min_score = float(candidate_content.min())
    max_score = float(candidate_content.max())

    if max_score > min_score:
        candidate_content = (
            (candidate_content - min_score)
            / (max_score - min_score)
        )
    else:
        candidate_content = np.zeros(len(candidates))

    cf_results = []
    hybrid_results = []

    for movie_id, content_score in zip(
        candidates, candidate_content
    ):
        rating_col = data["movie_to_rating_col"].get(movie_id)

        if (
            rating_col is not None
            and np.isfinite(cf_predictions[rating_col])
        ):
            predicted_rating = float(cf_predictions[rating_col])
        else:
            predicted_rating = overall_mean

        predicted_rating = float(
            np.clip(predicted_rating, 0.5, 5.0)
        )

        cf_normalized = float(
            np.clip((predicted_rating - 0.5) / 4.5, 0.0, 1.0)
        )

        hybrid_score = (
            CF_WEIGHT * cf_normalized
            + CONTENT_WEIGHT * float(content_score)
        )

        movie_row = data["movies"][
            data["movies"]["movieId"] == movie_id
        ]

        if movie_row.empty:
            continue

        movie = movie_row.iloc[0]

        result = {
            "movieId": int(movie_id),
            "title": str(movie["title"]),
            "genres": str(movie["genres"]),
            "predicted_rating": round(predicted_rating, 2),
            "score": round(float(hybrid_score), 4)
        }

        cf_result = result.copy()
        cf_result["score"] = round(predicted_rating, 4)

        cf_results.append(cf_result)
        hybrid_results.append(result)

    cf_results.sort(
        key=lambda item: item["score"], reverse=True
    )
    hybrid_results.sort(
        key=lambda item: item["score"], reverse=True
    )

    return cf_results[:limit], hybrid_results[:limit]


st.markdown(
    """
    <div class="hero">
        <h1>🎬 MovieLens - AI Recommender</h1>
        <p>Compare Collaborative Filtering with Hybrid Recommendations</p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    with st.spinner("Loading MovieLens data and building recommendation models..."):
        data = load_data()

    movies_df = data["movies"].drop_duplicates(
        subset=["movieId"]
    ).copy()

    ratings_df = data["ratings"]

    st.sidebar.title("🎬 MovieLens")
    st.sidebar.caption("Personalized Movie Recommendations")
    st.sidebar.divider()

    user_ids = sorted(data["user_ids"].tolist())

    selected_user = st.sidebar.selectbox(
        "Select MovieLens User ID", user_ids
    )

    user_ratings = ratings_df[
        ratings_df["userId"] == selected_user
    ]

    favorite_rated = user_ratings[
        user_ratings["rating"] >= 4.0
    ].sort_values("rating", ascending=False)

    default_ids = [
        int(mid)
        for mid in favorite_rated["movieId"].tolist()[:5]
        if int(mid) in data["movie_to_index"]
    ]

    movie_options = {
        f"{row.title} (ID: {int(row.movieId)})": int(row.movieId)
        for row in movies_df.itertuples(index=False)
    }

    default_labels = [
        label for label, movie_id in movie_options.items()
        if movie_id in default_ids
    ]

    selected_labels = st.sidebar.multiselect(
        "Choose favourite movies",
        options=list(movie_options.keys()),
        default=default_labels
    )

    selected_favorites = [
        movie_options[label] for label in selected_labels
    ]

    top_n = st.sidebar.slider(
        "Recommendations per method",
        min_value=5,
        max_value=20,
        value=10,
        step=5
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Movies", f"{len(movies_df):,}")
    col2.metric("Ratings", f"{len(ratings_df):,}")
    col3.metric("Users", f"{ratings_df['userId'].nunique():,}")

    st.subheader("Recommendation Settings")

    st.write(
        "The hybrid model combines 70% collaborative filtering "
        "with 30% content-based similarity. Movies already rated "
        "by the selected user are excluded."
    )

    if st.button("✨ Generate Recommendations"):
        try:
            with st.spinner("Generating recommendations..."):
                cf_results, hybrid_results = recommend(
                    data,
                    selected_user,
                    selected_favorites,
                    top_n
                )

            st.session_state["cf_results"] = cf_results
            st.session_state["hybrid_results"] = hybrid_results
            st.session_state["selected_user"] = selected_user

        except Exception as error:
            st.error(str(error))

    if "hybrid_results" in st.session_state:
        st.subheader("Recommendation Results")

        cf_tab, hybrid_tab = st.tabs([
            "Collaborative Filtering",
            "Hybrid Recommendation"
        ])

        def display_movies(results):
            if not results:
                st.warning("No recommendations available.")
                return

            for start in range(0, len(results), 2):
                columns = st.columns(2)

                for offset, container in enumerate(columns):
                    index = start + offset

                    if index >= len(results):
                        continue

                    movie = results[index]

                    title = html.escape(movie["title"])
                    genres = html.escape(movie["genres"])

                    with container:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div class="movie-title">{title}</div>
                                <div class="movie-genres">{genres}</div>
                                <div class="movie-rating">
                                    ⭐ Predicted rating:
                                    {movie['predicted_rating']:.2f}/5
                                </div>
                                <div>Score: {movie['score']:.4f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            results_df = pd.DataFrame(results)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download recommendations as CSV",
                data=results_df.to_csv(index=False).encode("utf-8"),
                file_name="movielens_recommendations.csv",
                mime="text/csv",
                key=f"download_{'cf' if results is st.session_state.get('cf_results') else 'hybrid'}"
            )

        with cf_tab:
            st.write(
                "Ranked using predicted ratings from similar users."
            )
            display_movies(st.session_state["cf_results"])

        with hybrid_tab:
            st.write(
                "Ranked using 70% collaborative filtering and "
                "30% content similarity."
            )
            display_movies(st.session_state["hybrid_results"])

    st.divider()

    st.caption(
        "MovieLens - AI Recommender | Python · Streamlit · "
        "Pandas · NumPy · Scikit-learn"
    )

except FileNotFoundError as error:
    st.error(
        "Dataset file not found. Make sure ratings.csv, movies.csv "
        "and tags.csv are inside the data folder."
    )
    st.exception(error)

except Exception as error:
    st.error("The recommendation app could not be loaded.")
    st.exception(error)
