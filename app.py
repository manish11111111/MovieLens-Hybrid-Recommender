from pathlib import Path
from html import escape

import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

CF_WEIGHT = 0.70
CONTENT_WEIGHT = 0.30
FAV_CF_WEIGHT = 0.50
FAV_CONTENT_WEIGHT = 0.50
FAVOURITE_RATING = 5.0
TOP_NEIGHBORS = 30
MIN_NEIGHBORS = 3
CF_SHRINKAGE = 1.0  # higher = predictions stay closer to the user's average
APP_NAME = "MovieLens"
AUTHOR = "Manish Chapagain"


st.set_page_config(
    page_title=f"{APP_NAME} • {AUTHOR}",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-primary: #090b10; --bg-secondary: #10131b; --bg-card: #141923;
    --text-primary: #f5f7fb; --text-secondary: #a4adbd; --text-tertiary: #778195;
    --accent: #ff5c5c; --accent-secondary: #ffb347;
    --border-subtle: rgba(255,255,255,0.09);
    --gradient: linear-gradient(110deg, #ff5c5c 0%, #ffb347 55%, #a78bfa 100%);
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(255,92,92,0.08), transparent 36%),
        radial-gradient(ellipse at 90% 12%, rgba(167,139,250,0.07), transparent 32%),
        var(--bg-primary);
    color: var(--text-primary);
}
[data-testid="stHeader"] { background: rgba(9,11,16,0.8); }
[data-testid="stSidebar"] { background: #0c0f16; border-right: 1px solid var(--border-subtle); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: var(--text-secondary); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; color: var(--text-primary); letter-spacing: -0.035em; }
p, li, label { color: var(--text-secondary); }
.hero {
    position: relative; overflow: hidden; padding: 2.4rem 2.3rem; margin: 0.2rem 0 1.8rem;
    border: 1px solid var(--border-subtle); border-radius: 24px;
    background:
        radial-gradient(circle at 90% 15%, rgba(167,139,250,0.16), transparent 30%),
        radial-gradient(circle at 75% 100%, rgba(255,92,92,0.12), transparent 35%),
        linear-gradient(135deg, #151a26 0%, #10131b 65%, #191420 100%);
}
.hero-eyebrow { color: #ffb347; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.19em; text-transform: uppercase; margin-bottom: 0.8rem; }
.hero-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(2.3rem, 5vw, 4.1rem); line-height: 1.03; font-weight: 700; letter-spacing: -0.065em; color: #ffffff; margin-bottom: 1rem; }
.gradient-text { background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-description { max-width: 690px; font-size: 1rem; line-height: 1.75; color: #aeb6c7; }
.section-kicker { font-size: 0.69rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-tertiary); margin-bottom: 0.35rem; }
.section-heading { font-family: 'Space Grotesk', sans-serif; color: var(--text-primary); font-size: 1.45rem; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 1rem; }
.stat-card { background: linear-gradient(145deg, #171c28, #10131b); border: 1px solid var(--border-subtle); border-radius: 17px; padding: 1.1rem 1.2rem; min-height: 112px; }
.stat-label { color: var(--text-tertiary); font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 0.6rem; }
.stat-value { color: #ffffff; font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.05em; }
.stat-note { color: var(--text-secondary); font-size: 0.76rem; margin-top: 0.25rem; }
.model-panel { padding: 1.2rem 1.4rem; margin: 1.2rem 0 1.6rem; border-radius: 16px; border: 1px solid var(--border-subtle); background: rgba(20,25,35,0.75); }
.model-title { font-family: 'Space Grotesk', sans-serif; color: #ffffff; font-size: 1rem; font-weight: 700; margin-bottom: 0.45rem; }
.model-description { color: var(--text-secondary); font-size: 0.85rem; line-height: 1.65; }
.model-pill { display: inline-block; border-radius: 999px; border: 1px solid rgba(255,179,71,0.28); background: rgba(255,179,71,0.08); color: #ffca7a; padding: 0.3rem 0.65rem; font-size: 0.69rem; font-weight: 700; margin: 0.25rem 0.25rem 0 0; }
.movie-card { height: 100%; min-height: 180px; padding: 1.15rem; margin-bottom: 0.8rem; background: linear-gradient(145deg, #171c28, #11141d); border: 1px solid var(--border-subtle); border-radius: 17px; transition: border-color 0.2s ease, transform 0.2s ease; }
.movie-card:hover { border-color: rgba(255,179,71,0.42); transform: translateY(-2px); }
.movie-rank { color: #ffb347; font-size: 0.67rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; margin-bottom: 0.55rem; }
.movie-title { color: #ffffff; font-family: 'Space Grotesk', sans-serif; font-size: 1.03rem; font-weight: 700; line-height: 1.4; margin-bottom: 0.55rem; overflow-wrap: anywhere; }
.movie-year { color: #aab3c4; font-size: 0.78rem; font-weight: 500; }
.movie-genres { color: #9ba5b7; font-size: 0.77rem; line-height: 1.6; margin: 0.65rem 0; }
.movie-score { display: inline-block; padding: 0.3rem 0.55rem; border-radius: 8px; background: rgba(255,92,92,0.11); color: #ff9292; font-size: 0.75rem; font-weight: 700; }
.movie-match { display: inline-block; padding: 0.3rem 0.55rem; border-radius: 8px; background: rgba(167,139,250,0.12); color: #c6b5ff; font-size: 0.75rem; font-weight: 700; margin-left: 0.3rem; }
.sidebar-brand { padding: 0.2rem 0 1.3rem; margin-bottom: 1rem; border-bottom: 1px solid var(--border-subtle); }
.sidebar-brand-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.06em; color: #ffffff; }
.sidebar-brand-subtitle { font-size: 0.72rem; color: var(--text-tertiary); margin-top: 0.2rem; }
.sidebar-credit { margin-top: 1.2rem; padding: 1rem 0.2rem 0.6rem; border-top: 1px solid var(--border-subtle); display: flex; flex-direction: column; gap: 0.45rem; }
.credit-label { color: var(--text-tertiary); font-family: 'Inter', sans-serif; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; }
.credit-name { width: fit-content; font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; letter-spacing: -0.035em; background: var(--gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
div.stButton > button { border: 1px solid rgba(255,179,71,0.35); border-radius: 11px; min-height: 2.8rem; color: #ffffff; background: linear-gradient(110deg, rgba(255,92,92,0.95), rgba(255,139,76,0.95)); font-family: 'DM Sans', sans-serif; font-weight: 700; transition: filter 0.2s ease, transform 0.2s ease; }
div.stButton > button:hover { color: #ffffff; filter: brightness(1.1); transform: translateY(-1px); border-color: rgba(255,179,71,0.7); }
div.stButton > button:focus { color: #ffffff; box-shadow: 0 0 0 2px rgba(255,179,71,0.22); }
[data-testid="stDownloadButton"] button { border-radius: 10px; border: 1px solid var(--border-subtle); background: #171c28; color: #f5f7fb; }
[data-testid="stTabs"] button { color: var(--text-secondary); font-weight: 600; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #ffb347; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div { background: #131823; border-color: var(--border-subtle); border-radius: 10px; }
[data-testid="stSlider"] [role="slider"] { background-color: #ffb347; }
.empty-state { padding: 2.2rem 1.4rem; text-align: center; background: rgba(20,25,35,0.6); border: 1px dashed rgba(255,255,255,0.15); border-radius: 17px; color: var(--text-secondary); }
.empty-state-title { color: #ffffff; font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
.app-footer { margin-top: 3rem; padding: 1.5rem 0 1rem; border-top: 1px solid var(--border-subtle); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.7rem; }
.footer-brand { font-family: 'Space Grotesk', sans-serif; color: #ffffff; font-weight: 700; font-size: 0.9rem; }
.footer-credit { color: var(--text-tertiary); font-size: 0.75rem; }
.footer-credit span { color: #ffb347; font-weight: 600; }
hr { border-color: var(--border-subtle); }
@media (max-width: 700px) {
    .hero { padding: 1.6rem 1.25rem; border-radius: 18px; }
    .hero-title { font-size: 2.35rem; }
    .stat-card { padding: 0.9rem; }
    .stat-value { font-size: 1.45rem; }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data + model (built ONCE, kept in memory, never copied between reruns)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and building the model...")
def load_model():
    ratings_path = DATA_DIR / "ratings.csv"
    movies_path = DATA_DIR / "movies.csv"
    tags_path = DATA_DIR / "tags.csv"

    if not ratings_path.exists():
        raise FileNotFoundError(f"Missing file: {ratings_path}")
    if not movies_path.exists():
        raise FileNotFoundError(f"Missing file: {movies_path}")

    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(movies_path)

    required_movies = {"movieId", "title", "genres"}
    if not required_movies.issubset(movies.columns):
        raise ValueError("movies.csv must contain movieId, title, and genres columns.")

    # ---- clean ratings
    for col in ["userId", "movieId", "rating"]:
        ratings[col] = pd.to_numeric(ratings[col], errors="coerce")
    ratings = ratings.dropna()
    ratings = ratings[ratings["rating"].between(0.5, 5.0)]
    ratings["userId"] = ratings["userId"].astype(np.int64)
    ratings["movieId"] = ratings["movieId"].astype(np.int64)
    ratings = ratings.drop_duplicates(subset=["userId", "movieId"], keep="last")

    # ---- clean movies
    movies["movieId"] = pd.to_numeric(movies["movieId"], errors="coerce")
    movies = movies.dropna(subset=["movieId", "title"]).copy()
    movies["movieId"] = movies["movieId"].astype(np.int64)
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].fillna("").astype(str)
    movies = movies.drop_duplicates(subset=["movieId"], keep="last").reset_index(drop=True)

    # ---- tags
    movies["tag_text"] = ""
    if tags_path.exists():
        tags = pd.read_csv(tags_path)
        if {"movieId", "tag"}.issubset(tags.columns):
            tags["movieId"] = pd.to_numeric(tags["movieId"], errors="coerce")
            tags = tags.dropna(subset=["movieId"])
            tags["movieId"] = tags["movieId"].astype(np.int64)
            tags["tag"] = tags["tag"].fillna("").astype(str)
            tag_text = tags.groupby("movieId")["tag"].agg(" ".join)
            movies["tag_text"] = movies["movieId"].map(tag_text).fillna("")

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0]
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["content_text"] = (
        movies["clean_title"]
        + " "
        + movies["genres"].str.replace("|", " ", regex=False)
        + " "
        + movies["tag_text"]
    ).str.lower()

    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    if ratings.empty:
        raise ValueError("No valid ratings match the movies dataset.")

    # ---- unique, human-friendly labels for the "add favourite" dropdown
    duplicated = movies["title"].duplicated(keep=False)
    labels = np.where(
        duplicated,
        movies["title"] + " [" + movies["movieId"].astype(str) + "]",
        movies["title"],
    )
    label_to_id = dict(zip(labels, movies["movieId"].astype(int)))
    label_order = np.argsort(movies["title"].str.lower().values, kind="stable")

    # ---- content features (sparse, L2-normalised rows)
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=50000
    )
    content_matrix = vectorizer.fit_transform(movies["content_text"]).tocsr()

    # ---- sparse user x movie matrices
    users = np.sort(ratings["userId"].unique())
    user_rows = np.searchsorted(users, ratings["userId"].values)
    movie_pos = pd.Series(np.arange(len(movies)), index=movies["movieId"].values)
    movie_cols = movie_pos.loc[ratings["movieId"].values].values

    shape = (len(users), len(movies))
    R = csr_matrix((ratings["rating"].values.astype(np.float64), (user_rows, movie_cols)), shape=shape)
    R.sort_indices()

    counts = np.diff(R.indptr)
    user_means = np.asarray(R.sum(axis=1)).ravel() / np.maximum(counts, 1)

    C = R.copy()  # mean-centred ratings (only on rated entries)
    C.data = C.data - np.repeat(user_means, counts)

    B = R.copy()  # binary "has rated" matrix
    B.data = np.ones_like(B.data)

    norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=1)).ravel())

    return {
        "ratings_count": int(len(ratings)),
        "avg_rating": float(ratings["rating"].mean()),
        "n_users": int(len(users)),
        "movies": movies,
        "titles": movies["title"].values,
        "genres": movies["genres"].values,
        "years": movies["year"].values,
        "movie_ids": movies["movieId"].values,
        "movie_id_to_index": {int(m): i for i, m in enumerate(movies["movieId"].values)},
        "labels": labels,
        "label_to_id": label_to_id,
        "label_order": label_order,
        "content_matrix": content_matrix,
        "R": R,
        "C": C,
        "B": B,
        "norms": norms,
        "user_means": user_means,
        "users": [int(u) for u in users],
        "user_to_row": {int(u): i for i, u in enumerate(users)},
        "global_mean": float(ratings["rating"].mean()),
    }


# --------------------------------------------------------------------------
# Recommender (sparse + vectorised, no Python loops over movies)
# --------------------------------------------------------------------------
def minmax(values):
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-9:
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


def build_frame(model, idx, predicted, score, cf_score=None, content_score=None):
    frame = pd.DataFrame({
        "movieId": model["movie_ids"][idx].astype(int),
        "title": model["titles"][idx],
        "genres": model["genres"][idx],
        "year": model["years"][idx],
        "predicted_rating": predicted[idx],
        "score": score[idx],
    })
    if cf_score is not None:
        frame["cf_score"] = cf_score[idx]
    if content_score is not None:
        frame["content_score"] = content_score[idx]
    return frame.reset_index(drop=True)


def recommend(model, user_id, favorite_ids, limit=10):
    n_movies = len(model["titles"])
    u = model["user_to_row"].get(int(user_id))
    empty = pd.DataFrame(columns=["movieId", "title", "genres", "year", "predicted_rating", "score"])

    if u is None:
        return empty, empty

    R, C, B = model["R"], model["C"], model["B"]
    norms = model["norms"]
    id_to_idx = model["movie_id_to_index"]
    user_mean = float(model["user_means"][u])

    start, end = R.indptr[u], R.indptr[u + 1]
    rated_idx = R.indices[start:end]
    rated_vals = R.data[start:end]

    fav_idx = list(dict.fromkeys(
        id_to_idx[int(m)] for m in favorite_ids if int(m) in id_to_idx
    ))

    valid = np.ones(n_movies, dtype=bool)
    valid[rated_idx] = False
    if fav_idx:
        valid[fav_idx] = False

    # ---- seed movies for content similarity
    seed_idx = list(fav_idx)
    if not seed_idx and len(rated_idx):
        order = np.argsort(-rated_vals, kind="stable")
        liked = order[rated_vals[order] >= 4.0][:10]
        if len(liked) == 0:
            liked = order[:5]
        seed_idx = rated_idx[liked].tolist()

    if seed_idx:
        profile = np.asarray(model["content_matrix"][seed_idx].mean(axis=0)).ravel()
        content = np.asarray(model["content_matrix"].dot(profile)).ravel()
    else:
        content = np.zeros(n_movies)

    # ---- user-based collaborative filtering
    cf_raw = np.full(n_movies, np.nan)

    # User profile = own centred ratings + favourites treated as top ratings
    profile = dict(zip(C.indices[start:end].tolist(), C.data[start:end].tolist()))
    for movie_index in fav_idx:
        profile[movie_index] = FAVOURITE_RATING - user_mean

    p_cols = np.fromiter(profile.keys(), dtype=np.int64, count=len(profile))
    p_vals = np.fromiter(profile.values(), dtype=np.float64, count=len(profile))
    p_vec = csr_matrix(
        (p_vals, (np.zeros(len(p_cols), dtype=np.int64), p_cols)),
        shape=(1, n_movies),
    )
    p_norm = float(np.sqrt((p_vals ** 2).sum()))

    dots = C.dot(p_vec.T).toarray().ravel()
    denom_norm = norms * p_norm
    sims = np.divide(dots, denom_norm, out=np.zeros_like(dots), where=denom_norm > 0)
    sims[u] = 0.0

    neighbors = np.flatnonzero(sims > 0)
    if len(neighbors) > TOP_NEIGHBORS:
        top = np.argpartition(-sims[neighbors], TOP_NEIGHBORS - 1)[:TOP_NEIGHBORS]
        neighbors = neighbors[top]

    if len(neighbors):
        weights = sims[neighbors]
        sub_c = C[neighbors]
        sub_b = B[neighbors]

        numerator = sub_c.T.dot(weights)
        denominator = sub_b.T.dot(weights)
        neighbor_count = np.asarray(sub_b.sum(axis=0)).ravel()

        ok = (denominator > 0) & (neighbor_count >= MIN_NEIGHBORS)
        cf_raw[ok] = user_mean + numerator[ok] / (denominator[ok] + CF_SHRINKAGE)

    cf_pred = np.clip(cf_raw, 0.5, 5.0)  # displayed rating (NaN stays NaN)
    cf_valid = valid & ~np.isnan(cf_raw)

    # ---- collaborative-only results
    cf_norm = np.zeros(n_movies)
    if cf_valid.any():
        cf_norm[cf_valid] = minmax(cf_raw[cf_valid])

    cf_candidates = np.flatnonzero(cf_valid)
    if len(cf_candidates):
        order = np.lexsort((-cf_pred[cf_candidates], -cf_raw[cf_candidates]))
        cf_top = cf_candidates[order][:limit]
        cf_results = build_frame(model, cf_top, cf_pred, cf_norm)
    else:
        cf_results = empty

    # ---- hybrid results
    content_norm = np.zeros(n_movies)
    if valid.any():
        content_norm[valid] = minmax(content[valid])

    if fav_idx:
        cf_w, content_w = FAV_CF_WEIGHT, FAV_CONTENT_WEIGHT
    else:
        cf_w, content_w = CF_WEIGHT, CONTENT_WEIGHT

    hybrid = cf_w * cf_norm + content_w * content_norm
    predicted = np.where(
        cf_valid,
        cf_pred,
        np.clip(user_mean + (content_norm - 0.5), 0.5, 5.0),
    )

    candidates = np.flatnonzero(valid & (cf_valid | (content > 0)))
    if len(candidates):
        order = np.lexsort((-predicted[candidates], -hybrid[candidates]))
        top = candidates[order][:limit]
        hybrid_results = build_frame(model, top, predicted, hybrid, cf_norm, content_norm)
    else:
        hybrid_results = empty

    return cf_results, hybrid_results


@st.cache_data(show_spinner=False, max_entries=256)
def cached_recommend(user_id, favorite_ids, limit):
    return recommend(load_model(), user_id, list(favorite_ids), limit)


# --------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------
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
            rating_text = "N/A" if pd.isna(rating) else f"{float(rating):.2f}/5"

            if is_hybrid:
                match_text = f"Match {float(movie.get('score', 0.0)) * 100:.0f}%"
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
            c for c in ["title", "year", "genres", "predicted_rating", "score", "cf_score", "content_score"]
            if c in rows.columns
        ]
        st.dataframe(rows[visible_columns], width="stretch", hide_index=True)

    st.download_button(
        label="Download recommendations as CSV",
        data=rows.to_csv(index=False).encode("utf-8"),
        file_name="movielens_recommendations.csv",
        mime="text/csv",
        key=download_key,
    )


def add_favourite():
    model = load_model()
    label = st.session_state.get("selected_movie_to_add", "")

    if label and label in model["label_to_id"]:
        movie_id = int(model["label_to_id"][label])
        current = list(st.session_state.get("favorite_ids", []))
        if movie_id not in current:
            current.append(movie_id)
            st.session_state["favorite_ids"] = current

    st.session_state["selected_movie_to_add"] = ""


def remove_favourite(movie_id):
    current = st.session_state.get("favorite_ids", [])
    st.session_state["favorite_ids"] = [m for m in current if m != movie_id]


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
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
    model = load_model()
    users = model["users"]
    id_to_idx = model["movie_id_to_index"]

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title"><span class="gradient-text">MovieLens</span></div>
            <div class="sidebar-brand-subtitle">Hybrid Recommendation System</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown('<div class="section-kicker">PERSONALIZE</div>', unsafe_allow_html=True)

    selected_user = st.sidebar.selectbox(
        "Select a user",
        options=users,
        format_func=lambda user_id: f"User {user_id}",
        key="selected_user",
    )

    if "favorite_ids" not in st.session_state:
        st.session_state["favorite_ids"] = []

    favorite_ids = [m for m in st.session_state["favorite_ids"] if m in id_to_idx]
    st.session_state["favorite_ids"] = favorite_ids

    st.sidebar.markdown('<div class="section-kicker">YOUR FAVOURITES</div>', unsafe_allow_html=True)

    if favorite_ids:
        for movie_id in favorite_ids:
            movie_title = model["titles"][id_to_idx[movie_id]]
            left, right = st.sidebar.columns([5, 1])

            with left:
                st.markdown(
                    f"<div style='font-size:0.8rem;padding:0.35rem 0;'>{escape(str(movie_title))}</div>",
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
        st.sidebar.caption("Add movies you enjoy to personalize your recommendations.")

    # Options: movies the selected user has not rated (fast numpy mask)
    user_row = model["user_to_row"][selected_user]
    R = model["R"]
    unrated_mask = np.ones(len(model["titles"]), dtype=bool)
    unrated_mask[R.indices[R.indptr[user_row]:R.indptr[user_row + 1]]] = False
    order = model["label_order"]
    option_labels = [""] + model["labels"][order[unrated_mask[order]]].tolist()

    if st.session_state.get("selected_movie_to_add", "") not in option_labels:
        st.session_state["selected_movie_to_add"] = ""

    st.sidebar.selectbox(
        "Find a movie to add",
        options=option_labels,
        key="selected_movie_to_add",
    )

    st.sidebar.button(
        "＋ Add favourite",
        on_click=add_favourite,
        width="stretch",
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

    st.markdown(
        """
        <div class="section-kicker">YOUR DATASET</div>
        <div class="section-heading">MovieLens at a glance</div>
        """,
        unsafe_allow_html=True,
    )

    stats = [
        ("MOVIES", f"{len(model['titles']):,}", "Titles in the catalogue"),
        ("USERS", f"{model['n_users']:,}", "Unique rating profiles"),
        ("RATINGS", f"{model['ratings_count']:,}", "Interactions analysed"),
        ("AVG. RATING", f"{model['avg_rating']:.2f}/5", "Across all ratings"),
    ]

    for column, (label, value, note) in zip(st.columns(4), stats):
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

    st.write("Choose a user, add favourite movies if you want, and generate a ranked list.")

    if st.button("✦ Generate recommendations", key="generate_recommendations", width="stretch"):
        st.session_state["recommendations_generated"] = True

    if st.session_state.get("recommendations_generated", False):
        with st.spinner("Analysing movie preferences and rating patterns..."):
            cf_results, hybrid_results = cached_recommend(
                int(selected_user),
                tuple(st.session_state["favorite_ids"]),
                int(recommendation_limit),
            )

        hybrid_tab, cf_tab = st.tabs(["✦ Hybrid recommendations", "◈ Collaborative filtering"])

        with hybrid_tab:
            st.caption(
                "Combines collaborative filtering and content similarity. "
                "With favourites added, both parts use them (50/50 weighting)."
            )
            display_movies(hybrid_results, download_key="download_hybrid", is_hybrid=True)

        with cf_tab:
            st.caption("Recommendations based on ratings from users with similar preferences.")
            display_movies(cf_results, download_key="download_cf", is_hybrid=False)
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
    st.info("Place ratings.csv, movies.csv, and optionally tags.csv inside the data folder.")

except Exception as error:
    st.error("The application could not finish loading.")
    st.exception(error)