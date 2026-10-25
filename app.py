from pathlib import Path
from html import escape
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CF_WEIGHT = 0.70
CONTENT_WEIGHT = 0.30
TOP_NEIGHBORS = 20
APP_NAME = "MovieLens"
AUTHOR = "Manish Chapagain"


st.set_page_config(
    page_title=f"{APP_NAME} • {AUTHOR}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

    <style>
    :root {
        --bg-primary: #090b10;
        --bg-secondary: #10131b;
        --bg-card: #141923;
        --bg-card-hover: #1a2030;
        --text-primary: #f5f7fb;
        --text-secondary: #a4adbd;
        --text-tertiary: #778195;
        --accent: #ff5c5c;
        --accent-secondary: #ffb347;
        --border-subtle: rgba(255,255,255,0.09);
        --gradient: linear-gradient(110deg, #ff5c5c 0%, #ffb347 55%, #a78bfa 100%);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(255,92,92,0.08), transparent 36%),
            radial-gradient(ellipse at 90% 12%, rgba(167,139,250,0.07), transparent 32%),
            var(--bg-primary);
        color: var(--text-primary);
    }

    [data-testid="stHeader"] {
        background: rgba(9,11,16,0.8);
    }

    [data-testid="stSidebar"] {
        background: #0c0f16;
        border-right: 1px solid var(--border-subtle);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary);
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text-primary);
        letter-spacing: -0.035em;
    }

    p, li, label {
        color: var(--text-secondary);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.3rem;
        margin: 0.2rem 0 1.8rem;
        border: 1px solid var(--border-subtle);
        border-radius: 24px;
        background:
            radial-gradient(circle at 90% 15%, rgba(167,139,250,0.16), transparent 30%),
            radial-gradient(circle at 75% 100%, rgba(255,92,92,0.12), transparent 35%),
            linear-gradient(135deg, #151a26 0%, #10131b 65%, #191420 100%);
    }

    .hero-eyebrow {
        color: #ffb347;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.19em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.3rem, 5vw, 4.1rem);
        line-height: 1.03;
        font-weight: 700;
        letter-spacing: -0.065em;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .gradient-text {
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .hero-description {
        max-width: 690px;
        font-size: 1rem;
        line-height: 1.75;
        color: #aeb6c7;
    }

    .section-kicker {
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--text-tertiary);
        margin-bottom: 0.35rem;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--text-primary);
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: linear-gradient(145deg, #171c28, #10131b);
        border: 1px solid var(--border-subtle);
        border-radius: 17px;
        padding: 1.1rem 1.2rem;
        min-height: 112px;
    }

    .stat-label {
        color: var(--text-tertiary);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.6rem;
    }

    .stat-value {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.05em;
    }

    .stat-note {
        color: var(--text-secondary);
        font-size: 0.76rem;
        margin-top: 0.25rem;
    }

    .model-panel {
        padding: 1.2rem 1.4rem;
        margin: 1.2rem 0 1.6rem;
        border-radius: 16px;
        border: 1px solid var(--border-subtle);
        background: rgba(20,25,35,0.75);
    }

    .model-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .model-description {
        color: var(--text-secondary);
        font-size: 0.85rem;
        line-height: 1.65;
    }

    .model-pill {
        display: inline-block;
        border-radius: 999px;
        border: 1px solid rgba(255,179,71,0.28);
        background: rgba(255,179,71,0.08);
        color: #ffca7a;
        padding: 0.3rem 0.65rem;
        font-size: 0.69rem;
        font-weight: 700;
        margin: 0.25rem 0.25rem 0 0;
    }

    .movie-card {
        height: 100%;
        min-height: 180px;
        padding: 1.15rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(145deg, #171c28, #11141d);
        border: 1px solid var(--border-subtle);
        border-radius: 17px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .movie-card:hover {
        border-color: rgba(255,179,71,0.42);
        transform: translateY(-2px);
    }

    .movie-rank {
        color: #ffb347;
        font-size: 0.67rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .movie-title {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.03rem;
        font-weight: 700;
        line-height: 1.4;
        margin-bottom: 0.55rem;
        overflow-wrap: anywhere;
    }

    .movie-year {
        color: #aab3c4;
        font-size: 0.78rem;
        font-weight: 500;
    }

    .movie-genres {
        color: #9ba5b7;
        font-size: 0.77rem;
        line-height: 1.6;
        margin: 0.65rem 0;
    }

    .movie-score {
        display: inline-block;
        padding: 0.3rem 0.55rem;
        border-radius: 8px;
        background: rgba(255,92,92,0.11);
        color: #ff9292;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .movie-match {
        display: inline-block;
        padding: 0.3rem 0.55rem;
        border-radius: 8px;
        background: rgba(167,139,250,0.12);
        color: #c6b5ff;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.3rem;
    }

    .sidebar-brand {
        padding: 0.2rem 0 1.3rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--border-subtle);
    }

    .sidebar-brand-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.06em;
        color: #ffffff;
    }

    .sidebar-brand-subtitle {
        font-size: 0.72rem;
        color: var(--text-tertiary);
        margin-top: 0.2rem;
    }

    .sidebar-credit {
        margin-top: 1.2rem;
        padding: 1rem 0.2rem 0.6rem;
        border-top: 1px solid var(--border-subtle);
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }

    .credit-label {
        color: var(--text-tertiary);
        font-family: 'Inter', sans-serif;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }

    .credit-name {
        width: fit-content;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.035em;
        background: var(--gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    div.stButton > button {
        border: 1px solid rgba(255,179,71,0.35);
        border-radius: 11px;
        min-height: 2.8rem;
        color: #ffffff;
        background: linear-gradient(110deg, rgba(255,92,92,0.95), rgba(255,139,76,0.95));
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        transition: filter 0.2s ease, transform 0.2s ease;
    }

    div.stButton > button:hover {
        color: #ffffff;
        filter: brightness(1.1);
        transform: translateY(-1px);
        border-color: rgba(255,179,71,0.7);
    }

    div.stButton > button:focus {
        color: #ffffff;
        box-shadow: 0 0 0 2px rgba(255,179,71,0.22);
    }

    [data-testid="stDownloadButton"] button {
        border-radius: 10px;
        border: 1px solid var(--border-subtle);
        background: #171c28;
        color: #f5f7fb;
    }

    [data-testid="stTabs"] button {
        color: var(--text-secondary);
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #ffb347;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #131823;
        border-color: var(--border-subtle);
        border-radius: 10px;
    }

    [data-testid="stSlider"] [role="slider"] {
        background-color: #ffb347;
    }

    .empty-state {
        padding: 2.2rem 1.4rem;
        text-align: center;
        background: rgba(20,25,35,0.6);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 17px;
        color: var(--text-secondary);
    }

    .empty-state-title {
        color: #ffffff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .app-footer {
        margin-top: 3rem;
        padding: 1.5rem 0 1rem;
        border-top: 1px solid var(--border-subtle);
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.7rem;
    }

    .footer-brand {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .footer-credit {
        color: var(--text-tertiary);
        font-size: 0.75rem;
    }

    .footer-credit span {
        color: #ffb347;
        font-weight: 600;
    }

    hr {
        border-color: var(--border-subtle);
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.6rem 1.25rem;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 2.35rem;
        }

        .stat-card {
            padding: 0.9rem;
        }

        .stat-value {
            font-size: 1.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    tags_path = DATA_DIR / "tags.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")

    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    if tags_path.exists():
        tags = pd.read_csv(tags_path)
    else:
        tags = pd.DataFrame(columns=["userId", "movieId", "tag"])

    required_ratings = {"userId", "movieId", "rating"}
    required_movies = {"movieId", "title", "genres"}

    if not required_ratings.issubset(ratings.columns):
        raise ValueError(
            "ratings.csv must contain userId, movieId, and rating columns."
        )

    if not required_movies.issubset(movies.columns):
        raise ValueError(
            "movies.csv must contain movieId, title, and genres columns."
        )

    ratings = ratings[["userId", "movieId", "rating"]].copy()
    movies = movies.copy()

    ratings["userId"] = pd.to_numeric(ratings["userId"], errors="coerce")
    ratings["movieId"] = pd.to_numeric(ratings["movieId"], errors="coerce")
    ratings["rating"] = pd.to_numeric(ratings["rating"], errors="coerce")
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    movies = movies.dropna(
        subset=["movieId", "title"]
    )

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)
    movies["movieId"] = movies["movieId"].astype(int)

    ratings = ratings[
        ratings["rating"].between(0.5, 5.0)
    ]

    ratings = ratings.drop_duplicates(
        subset=["userId", "movieId"],
        keep="last",
    )

    movies["title"] = movies["title"].fillna("Unknown title").astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)

    if not tags.empty and {"movieId", "tag"}.issubset(tags.columns):
        tags["movieId"] = pd.to_numeric(
            tags["movieId"], errors="coerce"
        )
        tags = tags.dropna(subset=["movieId"])
        tags["movieId"] = tags["movieId"].astype(int)
        tags["tag"] = tags["tag"].fillna("").astype(str)

        tag_text = (
            tags.groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values))
            .rename("tag_text")
            .reset_index()
        )

        movies = movies.merge(
            tag_text,
            on="movieId",
            how="left",
        )
    else:
        movies["tag_text"] = ""

    movies["tag_text"] = movies["tag_text"].fillna("")

    movies["year"] = movies["title"].str.extract(
        r"\((\d{4})\)\s*$"
    )

    movies["clean_title"] = movies["title"].str.replace(
        r"\s*\(\d{4}\)\s*$",
        "",
        regex=True,
    )

    movies["content_text"] = (
        movies["clean_title"].fillna("")
        + " "
        + movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["tag_text"]
    ).str.lower()

    movies = movies.drop_duplicates(
        subset=["movieId"],
        keep="last",
    ).reset_index(drop=True)

    valid_movie_ids = set(movies["movieId"])
    ratings = ratings[
        ratings["movieId"].isin(valid_movie_ids)
    ].copy()

    if ratings.empty:
        raise ValueError("No valid ratings match the movies dataset.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    content_matrix = vectorizer.fit_transform(
        movies["content_text"]
    )

    content_similarity = cosine_similarity(
        content_matrix,
        dense_output=False,
    )

    user_movie_matrix = ratings.pivot(
        index="userId",
        columns="movieId",
        values="rating",
    ).fillna(0)

    user_means = ratings.groupby("userId")["rating"].mean()

    centered_ratings = user_movie_matrix.copy().astype(float)

    for user_id in centered_ratings.index:
        rated_mask = centered_ratings.loc[user_id] > 0
        centered_ratings.loc[user_id, rated_mask] = (
            centered_ratings.loc[user_id, rated_mask]
            - user_means.loc[user_id]
        )

    user_similarity = cosine_similarity(centered_ratings)
    user_similarity = pd.DataFrame(
        user_similarity,
        index=centered_ratings.index,
        columns=centered_ratings.index,
    )

    movie_id_to_index = {
        movie_id: index
        for index, movie_id in enumerate(movies["movieId"])
    }

    movie_lookup = movies.set_index("movieId", drop=False)

    title_to_id = {
        row["title"]: int(row["movieId"])
        for _, row in movies.sort_values("title").iterrows()
    }

    return {
        "ratings": ratings,
        "movies": movies,
        "tags": tags,
        "user_movie_matrix": user_movie_matrix,
        "user_similarity": user_similarity,
        "content_matrix": content_matrix,
        "content_similarity": content_similarity,
        "movie_id_to_index": movie_id_to_index,
        "movie_lookup": movie_lookup,
        "title_to_id": title_to_id,
        "user_means": user_means,
        "users": sorted(ratings["userId"].unique().tolist()),
    }


def normalize_scores(scores):
    scores = pd.Series(scores, dtype=float)

    if scores.empty:
        return scores

    minimum = scores.min()
    maximum = scores.max()

    if maximum - minimum < 1e-9:
        if maximum > 0:
            return pd.Series(1.0, index=scores.index)
        return pd.Series(0.0, index=scores.index)

    return (scores - minimum) / (maximum - minimum)


def recommend(data, user_id, favorite_ids, limit=10):
    ratings = data["ratings"]
    movies = data["movies"]
    user_movie_matrix = data["user_movie_matrix"]
    user_similarity = data["user_similarity"]
    content_similarity = data["content_similarity"]
    movie_id_to_index = data["movie_id_to_index"]
    movie_lookup = data["movie_lookup"]
    user_means = data["user_means"]

    favorite_ids = list(
        dict.fromkeys(
            int(movie_id)
            for movie_id in favorite_ids
            if int(movie_id) in movie_id_to_index
        )
    )

    user_ratings = user_movie_matrix.loc[user_id]
    rated_movie_ids = set(
        user_ratings[user_ratings > 0].index.astype(int)
    )

    excluded_ids = rated_movie_ids.union(favorite_ids)

    if not favorite_ids:
        favorite_ids = (
            ratings[
                (ratings["userId"] == user_id)
                & (ratings["rating"] >= 4.0)
            ]
            .sort_values("rating", ascending=False)["movieId"]
            .head(10)
            .astype(int)
            .tolist()
        )

    if not favorite_ids:
        favorite_ids = (
            ratings[
                ratings["userId"] == user_id
            ]
            .sort_values("rating", ascending=False)["movieId"]
            .head(5)
            .astype(int)
            .tolist()
        )

    favorite_indices = [
        movie_id_to_index[movie_id]
        for movie_id in favorite_ids
        if movie_id in movie_id_to_index
    ]

    if favorite_indices:
        content_scores = np.asarray(
            content_similarity[favorite_indices].mean(axis=0)
        ).ravel()
    else:
        content_scores = np.zeros(len(movies))

    favorite_lookup_ids = set(favorite_ids)
    similar_users = user_similarity.loc[user_id].drop(
        labels=[user_id],
        errors="ignore",
    )

    similar_users = similar_users[
        similar_users > 0
    ].sort_values(ascending=False).head(TOP_NEIGHBORS)

    cf_scores = {}

    for movie_id in movies["movieId"].astype(int):
        if movie_id in excluded_ids:
            continue

        numerator = 0.0
        denominator = 0.0
        neighbor_count = 0

        for neighbor_id, similarity in similar_users.items():
            neighbor_rating = user_movie_matrix.at[
                neighbor_id, movie_id
            ] if movie_id in user_movie_matrix.columns else 0

            if neighbor_rating > 0:
                neighbor_mean = user_means.get(
                    neighbor_id,
                    ratings["rating"].mean(),
                )

                numerator += similarity * (
                    neighbor_rating - neighbor_mean
                )
                denominator += abs(similarity)
                neighbor_count += 1

        if denominator > 0 and neighbor_count >= 2:
            user_mean = user_means.get(
                user_id,
                ratings["rating"].mean(),
            )

            prediction = user_mean + numerator / denominator
            cf_scores[movie_id] = float(
                np.clip(prediction, 0.5, 5.0)
            )

    cf_series = pd.Series(cf_scores, dtype=float)

    if cf_series.empty:
        cf_results = pd.DataFrame(
            columns=[
                "movieId",
                "title",
                "genres",
                "year",
                "predicted_rating",
                "score",
            ]
        )
    else:
        cf_normalized = normalize_scores(cf_series)

        cf_results = pd.DataFrame({
            "movieId": cf_series.index.astype(int),
            "predicted_rating": cf_series.values,
            "score": cf_normalized.values,
        })

        cf_results = cf_results.merge(
            movies[["movieId", "title", "genres", "year"]],
            on="movieId",
            how="left",
        )

        cf_results = cf_results.sort_values(
            ["predicted_rating", "score"],
            ascending=False,
        ).head(limit)

    content_by_id = {
        movie_id: float(content_scores[index])
        for movie_id, index in movie_id_to_index.items()
        if movie_id not in excluded_ids
    }

    content_series = pd.Series(
        content_by_id,
        dtype=float,
    )

    cf_norm_all = normalize_scores(cf_series)
    content_norm_all = normalize_scores(content_series)

    all_candidate_ids = set(cf_norm_all.index).union(
        content_norm_all.index
    )

    hybrid_rows = []

    for movie_id in all_candidate_ids:
        cf_score = float(cf_norm_all.get(movie_id, 0.0))
        content_score = float(
            content_norm_all.get(movie_id, 0.0)
        )

        hybrid_score = (
            CF_WEIGHT * cf_score
            + CONTENT_WEIGHT * content_score
        )

        if movie_id not in movie_lookup.index:
            continue

        movie = movie_lookup.loc[movie_id]

        if isinstance(movie, pd.DataFrame):
            movie = movie.iloc[0]

        predicted_rating = float(
            np.clip(
                user_means.get(
                    user_id,
                    ratings["rating"].mean(),
                ) + (hybrid_score - 0.5) * 2.0,
                0.5,
                5.0,
            )
        )

        hybrid_rows.append({
            "movieId": int(movie_id),
            "title": movie["title"],
            "genres": movie["genres"],
            "year": movie["year"],
            "predicted_rating": predicted_rating,
            "score": hybrid_score,
            "content_score": content_score,
            "cf_score": cf_score,
        })

    hybrid_results = pd.DataFrame(hybrid_rows)

    if not hybrid_results.empty:
        hybrid_results = hybrid_results.sort_values(
            ["score", "predicted_rating"],
            ascending=False,
        ).head(limit)

    return cf_results, hybrid_results


def display_movies(results, download_key, is_hybrid=True):
    if results is None or results.empty:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">No recommendations yet</div>
                Try another user or add some favourite movies to generate recommendations.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    rows = results.reset_index(drop=True)

    for start in range(0, len(rows), 2):
        columns = st.columns(2)

        for offset, column in enumerate(columns):
            index = start + offset

            if index >= len(rows):
                continue

            movie = rows.iloc[index]

            title = escape(str(movie.get("title", "Unknown title")))
            genres = escape(str(movie.get("genres", "Unknown genres")))
            year = movie.get("year", "")

            if pd.isna(year) or str(year).strip() == "":
                year_text = "Year unavailable"
            else:
                year_text = escape(str(year))

            rating = movie.get("predicted_rating", np.nan)

            if pd.isna(rating):
                rating_text = "N/A"
            else:
                rating_text = f"{float(rating):.2f}/5"

            if is_hybrid:
                match_score = float(movie.get("score", 0.0))
                match_text = f"Match {match_score * 100:.0f}%"
            else:
                match_text = f"Rank #{index + 1}"

            with column:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div class="movie-rank">RECOMMENDATION {index + 1:02d}</div>
                        <div class="movie-title">{title}</div>
                        <div class="movie-year">{year_text}</div>
                        <div class="movie-genres">{genres}</div>
                        <span class="movie-score">Predicted {rating_text}</span>
                        <span class="movie-match">{match_text}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with st.expander("View recommendation data"):
        visible_columns = [
            column
            for column in [
                "title",
                "year",
                "genres",
                "predicted_rating",
                "score",
                "cf_score",
                "content_score",
            ]
            if column in rows.columns
        ]

        st.dataframe(
            rows[visible_columns],
            use_container_width=True,
            hide_index=True,
        )

    csv_data = rows.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download recommendations as CSV",
        data=csv_data,
        file_name="movielens_recommendations.csv",
        mime="text/csv",
        key=download_key,
        use_container_width=False,
    )


def add_favourite(label_to_id):
    selected_title = st.session_state.get("selected_movie_to_add")

    if selected_title and selected_title in label_to_id:
        movie_id = int(label_to_id[selected_title])
        current = st.session_state.get("favorite_ids", [])

        if movie_id not in current:
            st.session_state["favorite_ids"] = current + [movie_id]


def remove_favourite(movie_id):
    current = st.session_state.get("favorite_ids", [])
    st.session_state["favorite_ids"] = [
        item for item in current if item != movie_id
    ]


st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">PERSONALIZED MOVIE DISCOVERY</div>
        <div class="hero-title">
            Find your next<br>
            <span class="gradient-text">favorite movie.</span>
        </div>
        <div class="hero-description">
            Discover movies tailored to your taste with a hybrid recommendation
            engine combining collaborative filtering and content-based similarity.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


try:
    data = load_data()

    ratings = data["ratings"]
    movies = data["movies"]
    users = data["users"]
    title_to_id = data["title_to_id"]

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">
                <span class="gradient-text">MovieLens</span>
            </div>
            <div class="sidebar-brand-subtitle">
                Hybrid Recommendation System
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="section-kicker">PERSONALIZE</div>',
        unsafe_allow_html=True,
    )

    selected_user = st.sidebar.selectbox(
        "Select a user",
        options=users,
        format_func=lambda user_id: f"User {user_id}",
        key="selected_user",
    )

    if "favorite_ids" not in st.session_state:
        st.session_state["favorite_ids"] = []

    user_rated_ids = set(
        ratings.loc[
            ratings["userId"] == selected_user,
            "movieId",
        ].astype(int)
    )

    favorite_ids = st.session_state["favorite_ids"]
    valid_movie_ids = set(movies["movieId"].astype(int))

    favorite_ids = [
        movie_id
        for movie_id in favorite_ids
        if movie_id in valid_movie_ids
    ]

    st.session_state["favorite_ids"] = favorite_ids

    st.sidebar.markdown(
        '<div class="section-kicker">YOUR FAVOURITES</div>',
        unsafe_allow_html=True,
    )

    if favorite_ids:
        for movie_id in favorite_ids:
            movie_title = data["movie_lookup"].loc[movie_id, "title"]

            left, right = st.sidebar.columns([5, 1])

            with left:
                st.markdown(
                    f"<div style='font-size:0.8rem;padding:0.35rem 0;'>"
                    f"{escape(str(movie_title))}</div>",
                    unsafe_allow_html=True,
                )

            with right:
                st.button(
                    "×",
                    key=f"remove_{movie_id}",
                    on_click=remove_favourite,
                    args=(movie_id,),
                )
    else:
        st.sidebar.caption(
            "Add movies you enjoy to personalize your recommendations."
        )

    unrated_movies = movies[
        ~movies["movieId"].isin(user_rated_ids)
    ]

    movie_options = {
        row["title"]: int(row["movieId"])
        for _, row in unrated_movies.sort_values("title").iterrows()
    }

    st.sidebar.selectbox(
        "Find a movie to add",
        options=[""] + list(movie_options.keys()),
        key="selected_movie_to_add",
    )

    st.sidebar.button(
        "＋ Add favourite",
        on_click=add_favourite,
        args=(movie_options,),
        use_container_width=True,
    )

    recommendation_limit = st.sidebar.slider(
        "Number of recommendations",
        min_value=5,
        max_value=20,
        value=10,
        step=5,
    )

    st.sidebar.markdown(
        f"""
        <div class="sidebar-credit">
            <span class="credit-label">DESIGNED &amp; BUILT BY</span>
            <span class="credit-name">{escape(AUTHOR)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_movies = movies["movieId"].nunique()
    total_users = ratings["userId"].nunique()
    total_ratings = len(ratings)
    avg_rating = ratings["rating"].mean()

    st.markdown(
        """
        <div class="section-kicker">YOUR DATASET</div>
        <div class="section-heading">MovieLens at a glance</div>
        """,
        unsafe_allow_html=True,
    )

    stat_columns = st.columns(4)

    stats = [
        ("MOVIES", f"{total_movies:,}", "Titles in the catalogue"),
        ("USERS", f"{total_users:,}", "Unique rating profiles"),
        ("RATINGS", f"{total_ratings:,}", "Interactions analysed"),
        ("AVG. RATING", f"{avg_rating:.2f}/5", "Across all ratings"),
    ]

    for column, stat in zip(stat_columns, stats):
        label, value, note = stat

        with column:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"""
        <div class="model-panel">
            <div class="model-title">Hybrid recommendation engine</div>
            <div class="model-description">
                Collaborative filtering identifies patterns from similar users.
                Content-based filtering compares movie genres and available tags
                to estimate similarity to your preferences.
            </div>
            <span class="model-pill">Collaborative Filtering · {CF_WEIGHT:.0%}</span>
            <span class="model-pill">Content Similarity · {CONTENT_WEIGHT:.0%}</span>
            <span class="model-pill">Top Neighbours · {TOP_NEIGHBORS}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-kicker">DISCOVER</div>
        <div class="section-heading">Recommendations for you</div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "Choose a user, add favourite movies if you want, and generate a ranked list."
    )

    if st.button(
        "✦ Generate recommendations",
        key="generate_recommendations",
        use_container_width=True,
    ):
        st.session_state["recommendations_generated"] = True

    if st.session_state.get("recommendations_generated", False):
        with st.spinner("Analysing movie preferences and rating patterns..."):
            cf_results, hybrid_results = recommend(
                data,
                selected_user,
                st.session_state["favorite_ids"],
                recommendation_limit,
            )

        hybrid_tab, cf_tab = st.tabs([
            "✦ Hybrid recommendations",
            "◈ Collaborative filtering",
        ])

        with hybrid_tab:
            st.caption(
                "Combines collaborative filtering and content similarity "
                "using the configured model weights."
            )

            display_movies(
                hybrid_results,
                download_key="download_hybrid",
                is_hybrid=True,
            )

        with cf_tab:
            st.caption(
                "Recommendations based on ratings from users with similar preferences."
            )

            display_movies(
                cf_results,
                download_key="download_cf",
                is_hybrid=False,
            )

    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">Ready when you are</div>
                Generate recommendations to explore movies ranked for your selected user.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="app-footer">
            <div class="footer-brand">
                <span class="gradient-text">MovieLens</span>
                <span style="color:#778195;font-weight:400;"> · Hybrid Recommender</span>
            </div>
            <div class="footer-credit">
                Designed &amp; built by <span>{escape(AUTHOR)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

except FileNotFoundError as error:
    st.error("Required dataset files were not found.")
    st.code(str(error))
    st.info(
        "Place ratings.csv, movies.csv, and optionally tags.csv inside the data folder."
    )

except Exception as error:
    st.error("The application could not finish loading.")
    st.exception(error)