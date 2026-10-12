
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

# Rating bata user ko average rating ghataun
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

# Test data ma relevant movies chhanam
relevant_test = test_data[test_data['rating'] >= 4.0]

# Pratyek user ko relevant movies ko list banam
relevant_by_user = (
    relevant_test.groupby('userId')['movieId']
    .apply(set)
    .to_dict()
)

# Evaluation ko result store garam
precision_scores = []
recall_scores = []
evaluated_users = 0

# Test set ma relevant movie bhayeka users heram
for user_id, relevant_movies in relevant_by_user.items():

    # User training data ma chha ki heram
    if user_id not in ratings_matrix.index:
        continue

    # User le pahile rate nagareka movies nikalam
    unrated_movies = ratings_matrix.loc[user_id].isna()
    candidate_movies = ratings_matrix.columns[unrated_movies]

    # Relevant movie training ma already rate bhayeko rahecha bhane
    # tyo candidate list ma aaudaina
    if len(candidate_movies) == 0:
        continue

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

    # Similar users ko ratings nikalam
    neighbor_ratings = ratings_matrix.loc[
        similar_users.index
    ]

    # Rating ko average bata difference nikalam
    neighbor_means = user_means.loc[similar_users.index]

    deviations = neighbor_ratings.sub(
        neighbor_means,
        axis=0
    )

    # Similarity anusar weighted deviations calculate garam
    weighted_deviations = (
        deviations.fillna(0)
        .mul(similar_users, axis=0)
        .sum(axis=0)
    )

    # Movie rate gareka users ko similarity weights jodam
    available_weights = (
        neighbor_ratings.notna()
        .mul(similar_users, axis=0)
        .sum(axis=0)
    )

    # Predicted rating calculate garam
    predictions = (
        user_means.loc[user_id]
        + weighted_deviations /
        available_weights.replace(0, np.nan)
    )

    # Candidate movies matra rakhau
    predictions = predictions.reindex(candidate_movies)

    # Kamti ma 2 jana neighbors le rate gareka movies rakhau
    neighbor_counts = neighbor_ratings.notna().mul(
        similar_users.gt(0),
        axis=0
    ).sum(axis=0)

    predictions = predictions[
        neighbor_counts.reindex(predictions.index).fillna(0) >= 2
    ].dropna()

    # Highest predicted ratings bhayeka top 10 chhanam
    top_10_movies = set(
        predictions.sort_values(ascending=False)
        .head(10)
        .index
    )

    # Recommended movies madhye kati relevant chhan ganam
    hits = len(top_10_movies.intersection(relevant_movies))

    # Precision@10 calculate garam
    precision_at_10 = hits / 10

    # Recall@10 calculate garam
    recall_at_10 = hits / len(relevant_movies)

    precision_scores.append(precision_at_10)
    recall_scores.append(recall_at_10)
    evaluated_users += 1

# Average metrics calculate garam
print("\nRanking Evaluation Results")
print("---------------------------")
print(f"Users evaluated: {evaluated_users}")

if evaluated_users > 0:
    print(f"Precision@10: {np.mean(precision_scores):.4f}")
    print(f"Recall@10: {np.mean(recall_scores):.4f}")
else:
    print("No users could be evaluated with the current criteria.")
