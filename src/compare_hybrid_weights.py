
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

# TF-IDF features banam
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
user_similarity = cosine_similarity(centered)
np.fill_diagonal(user_similarity, 0)

user_ids = user_movie.index.to_numpy()
user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

rating_movie_ids = user_movie.columns.to_numpy()

# Training ma man pareka movies khojam
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

# Evaluation results store garam
results = {
    "CF": [],
    "Hybrid 70/30": [],
    "Hybrid 50/50": [],
    "Hybrid 30/70": [],
    "Hybrid 20/80": [],
    "Hybrid 10/90": []
}

print("Starting fair hybrid weight comparison...")

# Same users ra candidate set use garam
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

    # Candidate movies banam
    candidates = [
        movie_id
        for movie_id in movie_ids
        if movie_id not in seen_movies
    ]

    if not candidates:
        continue

    # Content profile banam
    liked_indices = [
        movie_to_index[mid]
        for mid in liked_movies
        if mid in movie_to_index
    ]

    if not liked_indices:
        continue

    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    content_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).ravel()

    # Candidate content scores nikalam
    candidate_indices = np.array([
        movie_to_index[mid] for mid in candidates
    ])

    candidate_content = content_scores[candidate_indices]

    # Content scores normalize garam
    content_min = candidate_content.min()
    content_max = candidate_content.max()

    if content_max > content_min:
        candidate_content = (
            (candidate_content - content_min)
            / (content_max - content_min)
        )
    else:
        candidate_content = np.zeros(len(candidates))

    content_by_movie = dict(zip(candidates, candidate_content))

    # Similar users select garam
    similarities = user_similarity[user_index]
    neighbor_indices = np.argsort(similarities)[::-1][:20]

    neighbor_indices = [
        idx for idx in neighbor_indices
        if similarities[idx] > 0
    ]

    if not neighbor_indices:
        continue

    # Neighbor ratings bata CF scores calculate garam
    weights = similarities[neighbor_indices]
    neighbor_ratings = rating_array[neighbor_indices]
    neighbor_masks = rating_mask[neighbor_indices]

    deviations = np.where(
        neighbor_masks,
        neighbor_ratings - user_means[neighbor_indices, None],
        0
    )

    weighted_sum = (weights[:, None] * deviations).sum(axis=0)
    weight_sum = (weights[:, None] * neighbor_masks).sum(axis=0)

    cf_predictions = np.full(
        len(rating_movie_ids),
        np.nan,
        dtype=np.float32
    )

    valid = weight_sum > 0

    cf_predictions[valid] = (
        user_means[user_index]
        + weighted_sum[valid] / weight_sum[valid]
    )

    # CF scores lai movie ID sanga map garam
    cf_by_movie = {}

    for col, movie_id in enumerate(rating_movie_ids):
        if movie_id in seen_movies:
            continue

        if np.isfinite(cf_predictions[col]):
            cf_by_movie[movie_id] = float(cf_predictions[col])

    # Sabai models ko common candidates banam
    common_candidates = [
        movie_id
        for movie_id in candidates
        if movie_id in cf_by_movie
    ]

    if len(common_candidates) < 10:
        continue

    cf_values = np.array([
        cf_by_movie[mid] for mid in common_candidates
    ])

    content_values = np.array([
        content_by_movie[mid] for mid in common_candidates
    ])

    # CF scores lai 0-1 range ma normalize garam
    cf_values = np.clip((cf_values - 0.5) / 4.5, 0, 1)

    # Different weights compare garam
    score_sets = {
        "CF": cf_values,
        "Hybrid 70/30": 0.7 * cf_values + 0.3 * content_values,
        "Hybrid 50/50": 0.5 * cf_values + 0.5 * content_values,
        "Hybrid 30/70": 0.3 * cf_values + 0.7 * content_values,
        "Hybrid 20/80": 0.2 * cf_values + 0.8 * content_values,
        "Hybrid 10/90": 0.1 * cf_values + 0.9 * content_values
    }

    # Precision ra recall calculate garam
    for model_name, scores in score_sets.items():

        top_indices = np.argsort(scores)[::-1][:10]

        recommended = {
            common_candidates[index]
            for index in top_indices
        }

        hits = len(recommended & relevant_movies)

        precision = hits / 10
        recall = hits / len(relevant_movies)

        results[model_name].append((precision, recall))

# Final results print garam
print("\nFair Hybrid Weight Comparison")
print("--------------------------------")

for model_name, metrics in results.items():

    if metrics:
        precision = np.mean([item[0] for item in metrics])
        recall = np.mean([item[1] for item in metrics])

        print(
            f"{model_name:15} | "
            f"Precision@10: {precision:.4f} | "
            f"Recall@10: {recall:.4f}"
        )
    else:
        print(f"{model_name:15} | No evaluation results")

print(f"\nUsers evaluated: {len(results['CF'])}")
