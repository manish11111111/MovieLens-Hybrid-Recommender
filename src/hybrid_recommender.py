
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")
tags = pd.read_csv("data/tags.csv")

# Train ra test data chhutyam
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Movie ko tags jodam
tags_by_movie = (
    tags.groupby("movieId")["tag"]
    .apply(lambda values: " ".join(values.astype(str)))
    .reset_index()
)

movies = movies.merge(tags_by_movie, on="movieId", how="left")
movies["tag"] = movies["tag"].fillna("")

# Genre ra tags combine garam
movies["content"] = (
    movies["genres"].str.replace("|", " ", regex=False)
    + " "
    + movies["tag"]
)

# Sparse TF-IDF features banam
tfidf = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32
)
tfidf_matrix = tfidf.fit_transform(movies["content"])

# Movie ID mapping banam
movie_ids = movies["movieId"].to_numpy()
movie_to_index = {
    movie_id: index
    for index, movie_id in enumerate(movie_ids)
}

# Training ratings matrix banam
user_movie = train_data.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Rating matrix lai array ma convert garam
rating_array = user_movie.to_numpy(dtype=np.float32)
rating_mask = np.isfinite(rating_array)

# User averages calculate garam
user_means = np.nanmean(rating_array, axis=1)
centered = np.where(
    rating_mask,
    rating_array - user_means[:, None],
    0
)

# User similarity calculate garam
user_similarity_array = cosine_similarity(centered)
np.fill_diagonal(user_similarity_array, 0)

user_ids = user_movie.index.to_numpy()
user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

# Movie columns ko mapping banam
rating_movie_ids = user_movie.columns.to_numpy()
movie_to_rating_col = {
    movie_id: index
    for index, movie_id in enumerate(rating_movie_ids)
}

# User ko training ma man pareka movies khojam
liked_train = train_data[train_data["rating"] >= 4.0]
liked_by_user = (
    liked_train.groupby("userId")["movieId"]
    .apply(list)
    .to_dict()
)

# Test ma relevant movies khojam
relevant_test = test_data[test_data["rating"] >= 4.0]
relevant_by_user = (
    relevant_test.groupby("userId")["movieId"]
    .apply(set)
    .to_dict()
)

# Results store garam
cf_precision = []
cf_recall = []
hybrid_precision = []
hybrid_recall = []

# Common users evaluate garam
print("Starting hybrid evaluation...")

for user_id, relevant_movies in relevant_by_user.items():

    if user_id not in user_to_index:
        continue

    liked_movies = liked_by_user.get(user_id, [])
    if not liked_movies:
        continue

    user_index = user_to_index[user_id]

    # Training ma rate gareka movies exclude garam
    seen_movies = set(
        train_data.loc[
            train_data["userId"] == user_id,
            "movieId"
        ]
    )

    # Content profile ko score calculate garam
    liked_indices = [
        movie_to_index[mid]
        for mid in liked_movies
        if mid in movie_to_index
    ]

    if not liked_indices:
        continue

    # Liked movies ko TF-IDF average profile banam
    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    # Sparse features sanga similarity calculate garam
    content_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).ravel()

    # Collaborative filtering ko neighbor select garam
    similarities = user_similarity_array[user_index]
    neighbor_indices = np.argsort(similarities)[::-1][:20]

    # Positive similarity bhayeka neighbors matra rakhau
    neighbor_indices = [
        idx for idx in neighbor_indices
        if similarities[idx] > 0
    ]

    if len(neighbor_indices) == 0:
        continue

    # Neighbor ko ratings bata CF score calculate garam
    neighbor_weights = similarities[neighbor_indices]
    neighbor_ratings = rating_array[neighbor_indices]
    neighbor_masks = rating_mask[neighbor_indices]

    deviations = np.where(
        neighbor_masks,
        neighbor_ratings - user_means[neighbor_indices, None],
        0
    )

    weighted_sum = (neighbor_weights[:, None] * deviations).sum(axis=0)
    weight_sum = (
        neighbor_weights[:, None] * neighbor_masks
    ).sum(axis=0)

    cf_predictions = np.full(
        len(rating_movie_ids),
        np.nan,
        dtype=np.float32
    )

    valid_columns = weight_sum > 0
    cf_predictions[valid_columns] = (
        user_means[user_index]
        + weighted_sum[valid_columns] / weight_sum[valid_columns]
    )

    # Candidate movies ko list banam
    candidates = [
        movie_id
        for movie_id in movie_ids
        if movie_id not in seen_movies
    ]

    if not candidates:
        continue

    # Content scores lai candidate set ma normalize garam
    candidate_indices = np.array([
        movie_to_index[mid] for mid in candidates
    ])

    candidate_content = content_scores[candidate_indices]
    min_score = candidate_content.min()
    max_score = candidate_content.max()

    if max_score > min_score:
        candidate_content = (
            (candidate_content - min_score)
            / (max_score - min_score)
        )
    else:
        candidate_content = np.zeros(len(candidates))

    # CF score lai candidate movies ma map garam
    cf_score_by_movie = {}

    for col_index, movie_id in enumerate(rating_movie_ids):
        if movie_id in seen_movies:
            continue

        if np.isfinite(cf_predictions[col_index]):
            cf_score_by_movie[movie_id] = float(
                cf_predictions[col_index]
            )

    # CF ra hybrid ko candidate ranking banam
    cf_rank_scores = {}
    hybrid_scores = {}

    for movie_id, content_score in zip(
        candidates, candidate_content
    ):
        if movie_id not in cf_score_by_movie:
            continue

        cf_score = cf_score_by_movie[movie_id]

        # CF prediction lai 0-1 range ma normalize garam
        cf_normalized = np.clip((cf_score - 0.5) / 4.5, 0, 1)

        cf_rank_scores[movie_id] = cf_score

        # Hybrid score calculate garam
        hybrid_scores[movie_id] = (
            0.7 * cf_normalized
            + 0.3 * content_score
        )

    if not cf_rank_scores or not hybrid_scores:
        continue

    # Top 10 recommendations nikalam
    cf_top10 = sorted(
        cf_rank_scores,
        key=cf_rank_scores.get,
        reverse=True
    )[:10]

    hybrid_top10 = sorted(
        hybrid_scores,
        key=hybrid_scores.get,
        reverse=True
    )[:10]

    # Precision ra recall calculate garam
    cf_hits = len(set(cf_top10) & relevant_movies)
    hybrid_hits = len(set(hybrid_top10) & relevant_movies)

    cf_precision.append(cf_hits / 10)
    cf_recall.append(cf_hits / len(relevant_movies))

    hybrid_precision.append(hybrid_hits / 10)
    hybrid_recall.append(hybrid_hits / len(relevant_movies))

# Final results print garam
print("\nCollaborative Filtering vs Hybrid")
print("---------------------------------")
print(f"Users evaluated: {len(hybrid_precision)}")

if hybrid_precision:
    print(
        f"Collaborative Filtering Precision@10: "
        f"{np.mean(cf_precision):.4f}"
    )
    print(
        f"Collaborative Filtering Recall@10: "
        f"{np.mean(cf_recall):.4f}"
    )
    print(
        f"Hybrid Precision@10: "
        f"{np.mean(hybrid_precision):.4f}"
    )
    print(
        f"Hybrid Recall@10: "
        f"{np.mean(hybrid_recall):.4f}"
    )
else:
    print("No users could be evaluated.")
