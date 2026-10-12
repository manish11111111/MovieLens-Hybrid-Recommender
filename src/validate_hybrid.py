
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv("data/ratings.csv")
movies = pd.read_csv("data/movies.csv")

# Train, validation ra test data chhutyam
train_data, remaining_data = train_test_split(
    ratings,
    test_size=0.30,
    random_state=42
)

validation_data, test_data = train_test_split(
    remaining_data,
    test_size=2 / 3,
    random_state=42
)

print("Dataset Split")
print("-------------")
print(f"Training ratings: {len(train_data)}")
print(f"Validation ratings: {len(validation_data)}")
print(f"Test ratings: {len(test_data)}")

# Movie genres bata content features banam
movies["content"] = (
    movies["genres"]
    .fillna("")
    .str.replace("|", " ", regex=False)
    .str.replace("(no genres listed)", "", regex=False)
)

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

# Sabai movies ko columns include garam
user_movie = user_movie.reindex(columns=movie_ids)

user_ids = user_movie.index.to_numpy()

user_to_index = {
    user_id: index
    for index, user_id in enumerate(user_ids)
}

rating_array = user_movie.to_numpy(dtype=np.float32)
rating_mask = np.isfinite(rating_array)

# User ko average rating calculate garam
user_means = np.nanmean(rating_array, axis=1)

centered = np.where(
    rating_mask,
    rating_array - user_means[:, None],
    0
)

# User similarity calculate garam
user_similarity = cosine_similarity(centered)
np.fill_diagonal(user_similarity, 0)

# Training ma man pareka movies khojam
liked_train = train_data[train_data["rating"] >= 4.0]

liked_by_user = (
    liked_train.groupby("userId")["movieId"]
    .apply(list)
    .to_dict()
)

# Test/validation ma relevant movies khojam
def get_relevant_movies(data):
    relevant_data = data[data["rating"] >= 4.0]

    return (
        relevant_data.groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )


# Ek user ko CF ra content scores calculate garam
def get_user_scores(user_id):

    if user_id not in user_to_index:
        return None

    liked_movies = liked_by_user.get(user_id, [])

    if not liked_movies:
        return None

    user_index = user_to_index[user_id]

    # Training ma rate gareka movies exclude garam
    seen_movies = set(
        train_data.loc[
            train_data["userId"] == user_id,
            "movieId"
        ]
    )

    candidates = [
        movie_id
        for movie_id in movie_ids
        if movie_id not in seen_movies
    ]

    if len(candidates) < 10:
        return None

    # User ko content profile banam
    liked_indices = [
        movie_to_index[movie_id]
        for movie_id in liked_movies
        if movie_id in movie_to_index
    ]

    if not liked_indices:
        return None

    user_profile = tfidf_matrix[liked_indices].mean(axis=0)
    user_profile = np.asarray(user_profile)

    # Candidate movies ko content similarity calculate garam
    content_scores = cosine_similarity(
        user_profile,
        tfidf_matrix
    ).ravel()

    candidate_indices = np.array([
        movie_to_index[movie_id]
        for movie_id in candidates
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

    content_by_movie = dict(
        zip(candidates, candidate_content)
    )

    # Similar users select garam
    similarities = user_similarity[user_index]

    neighbor_indices = np.argsort(similarities)[::-1][:20]

    neighbor_indices = [
        index
        for index in neighbor_indices
        if similarities[index] > 0
    ]

    # CF prediction calculate garam
    cf_predictions = np.full(
        len(movie_ids),
        user_means[user_index],
        dtype=np.float32
    )

    if neighbor_indices:

        weights = similarities[neighbor_indices]

        neighbor_ratings = rating_array[neighbor_indices]
        neighbor_masks = rating_mask[neighbor_indices]

        deviations = np.where(
            neighbor_masks,
            neighbor_ratings - user_means[neighbor_indices, None],
            0
        )

        weighted_sum = (
            weights[:, None] * deviations
        ).sum(axis=0)

        weight_sum = (
            weights[:, None] * neighbor_masks
        ).sum(axis=0)

        valid = weight_sum > 0

        cf_predictions[valid] = (
            user_means[user_index]
            + weighted_sum[valid] / weight_sum[valid]
        )

    # Candidate movies ko CF scores normalize garam
    cf_values = np.array([
        cf_predictions[movie_to_index[movie_id]]
        for movie_id in candidates
    ])

    cf_values = np.clip(
        (cf_values - 0.5) / 4.5,
        0,
        1
    )

    cf_by_movie = dict(zip(candidates, cf_values))

    return candidates, cf_by_movie, content_by_movie


# Ranking metrics calculate garam
def calculate_metrics(recommendations, relevant_movies):

    hits = len(set(recommendations) & relevant_movies)

    precision = hits / 10

    if len(relevant_movies) > 0:
        recall = hits / len(relevant_movies)
    else:
        recall = 0.0

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return precision, recall, f1


# Euta evaluation dataset ma model test garam
def evaluate_models(evaluation_data, model_weights=None):

    relevant_by_user = get_relevant_movies(evaluation_data)

    model_results = {
        "CF": [],
    }

    if model_weights is not None:
        for name in model_weights:
            model_results[name] = []

    users_evaluated = 0

    for user_id, relevant_movies in relevant_by_user.items():

        user_result = get_user_scores(user_id)

        if user_result is None:
            continue

        candidates, cf_scores, content_scores = user_result

        # Sabai models ko same candidate set use garam
        users_evaluated += 1

        # CF ranking calculate garam
        cf_ranked = sorted(
            candidates,
            key=lambda movie_id: cf_scores[movie_id],
            reverse=True
        )[:10]

        model_results["CF"].append(
            calculate_metrics(cf_ranked, relevant_movies)
        )

        # Different hybrid weights test garam
        if model_weights is not None:

            for model_name, cf_weight in model_weights.items():

                content_weight = 1 - cf_weight

                hybrid_scores = {
                    movie_id: (
                        cf_weight * cf_scores[movie_id]
                        + content_weight * content_scores[movie_id]
                    )
                    for movie_id in candidates
                }

                hybrid_ranked = sorted(
                    candidates,
                    key=lambda movie_id: hybrid_scores[movie_id],
                    reverse=True
                )[:10]

                model_results[model_name].append(
                    calculate_metrics(
                        hybrid_ranked,
                        relevant_movies
                    )
                )

    # Results summarize garam
    summary = {}

    for model_name, metrics in model_results.items():

        if metrics:
            summary[model_name] = {
                "precision": np.mean(
                    [item[0] for item in metrics]
                ),
                "recall": np.mean(
                    [item[1] for item in metrics]
                ),
                "f1": np.mean(
                    [item[2] for item in metrics]
                )
            }

    return summary, users_evaluated


# Validation ma weight select garam
weights_to_test = {
    "Hybrid 70/30": 0.7,
    "Hybrid 50/50": 0.5,
    "Hybrid 30/70": 0.3,
    "Hybrid 20/80": 0.2,
    "Hybrid 10/90": 0.1
}

print("\nEvaluating models on validation data...")
validation_results, validation_users = evaluate_models(
    validation_data,
    weights_to_test
)

print("\nValidation Results")
print("------------------")
print(f"Users evaluated: {validation_users}")

for model_name, scores in validation_results.items():
    print(
        f"{model_name:15} | "
        f"Precision@10: {scores['precision']:.4f} | "
        f"Recall@10: {scores['recall']:.4f} | "
        f"F1@10: {scores['f1']:.4f}"
    )

# Validation F1 bata best hybrid weight select garam
hybrid_names = list(weights_to_test.keys())

available_hybrids = [
    name for name in hybrid_names
    if name in validation_results
]

if not available_hybrids:
    raise RuntimeError(
        "No hybrid results available. Check the training data."
    )

best_model = max(
    available_hybrids,
    key=lambda name: validation_results[name]["f1"]
)

best_cf_weight = weights_to_test[best_model]

print(f"\nSelected model: {best_model}")
print(f"Selected CF weight: {best_cf_weight:.0%}")
print(f"Selected content weight: {1 - best_cf_weight:.0%}")

# Selected weight matra final test ma evaluate garam
print("\nEvaluating selected model on untouched test data...")

test_weights = {
    best_model: best_cf_weight
}

test_results, test_users = evaluate_models(
    test_data,
    test_weights
)

print("\nFinal Test Results")
print("------------------")
print(f"Users evaluated: {test_users}")

for model_name, scores in test_results.items():
    print(
        f"{model_name:15} | "
        f"Precision@10: {scores['precision']:.4f} | "
        f"Recall@10: {scores['recall']:.4f} | "
        f"F1@10: {scores['f1']:.4f}"
    )
