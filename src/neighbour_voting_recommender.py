
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# Training ra testing data banam
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# User ra movie ko rating matrix banam
ratings_matrix = train_data.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# User ko average rating nikalam
user_means = ratings_matrix.mean(axis=1)

# Rating bata user ko average ghataun
centered_ratings = ratings_matrix.sub(
    user_means,
    axis=0
).fillna(0)

# Users bich similarity calculate garam
similarity_matrix = cosine_similarity(centered_ratings)

user_similarity = pd.DataFrame(
    similarity_matrix,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

# Relevant movies ko list banam
relevant_test = test_data[test_data['rating'] >= 4.0]

relevant_by_user = (
    relevant_test.groupby('userId')['movieId']
    .apply(set)
    .to_dict()
)

# Evaluation ko result store garam
precision_scores = []
recall_scores = []
evaluated_users = 0

# Pratyek user ko recommendations calculate garam
for user_id, relevant_movies in relevant_by_user.items():

    # User training data ma chha ki heram
    if user_id not in ratings_matrix.index:
        continue

    # User le rate nagareka movies chhanam
    unrated_mask = ratings_matrix.loc[user_id].isna()
    candidate_movies = ratings_matrix.columns[unrated_mask]

    # Similar users nikalam
    similar_users = (
        user_similarity[user_id]
        .drop(user_id)
        .sort_values(ascending=False)
    )

    # Positive similarity bhayeka top 20 users chhanam
    similar_users = similar_users[
        similar_users > 0
    ].head(20)

    if similar_users.empty:
        continue

    # Similar users ko ratings nikalam
    neighbor_ratings = ratings_matrix.loc[
        similar_users.index
    ]

    # Similarity weights lai apply garam
    positive_ratings = neighbor_ratings.clip(lower=0)

    # Rating 4 wa mathi bhayeko vote matra rakhau
    liked_movies = neighbor_ratings >= 4.0

    # Similarity-weighted votes calculate garam
    weighted_votes = (
        liked_movies.mul(similar_users, axis=0)
        .sum(axis=0)
    )

    # Movie rate gareka similar users ko total weight nikalam
    available_weights = (
        neighbor_ratings.notna()
        .mul(similar_users, axis=0)
        .sum(axis=0)
    )

    # Weighted vote lai available weight le divide garam
    recommendation_scores = (
        weighted_votes /
        available_weights.replace(0, np.nan)
    )

    # User le pahile rate nagareka movies matra rakhau
    recommendation_scores = recommendation_scores[
        unrated_mask
    ].dropna()

    # Kamti ma 2 jana similar users le rate gareka movies rakhau
    neighbor_counts = neighbor_ratings.notna().mul(
        similar_users.gt(0),
        axis=0
    ).sum(axis=0)

    recommendation_scores = recommendation_scores[
        neighbor_counts.loc[recommendation_scores.index] >= 2
    ]

    # Highest vote score bhayeka top 10 movies chhanam
    top_10_movies = set(
        recommendation_scores.sort_values(
            ascending=False
        ).head(10).index
    )

    # Relevant recommendations ko count nikalam
    hits = len(top_10_movies.intersection(relevant_movies))

    # Precision ra recall calculate garam
    precision_scores.append(hits / 10)
    recall_scores.append(hits / len(relevant_movies))
    evaluated_users += 1

# Average results print garam
print("\nNeighbor-Voting Ranking Results")
print("--------------------------------")
print(f"Users evaluated: {evaluated_users}")

if evaluated_users > 0:
    print(f"Precision@10: {np.mean(precision_scores):.4f}")
    print(f"Recall@10: {np.mean(recall_scores):.4f}")
else:
    print("No users could be evaluated.")
