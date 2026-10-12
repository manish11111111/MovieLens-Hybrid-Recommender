
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')

# User ra movie ko rating table banam
ratings_matrix = ratings.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# Pratyek user ko average rating nikalam
user_means = ratings_matrix.mean(axis=1)

# User ko average bata rating ko difference nikalam
centered_ratings = ratings_matrix.sub(user_means, axis=0)

# Rating nadeko thau lai similarity calculate garna 0 banam
centered_ratings_filled = centered_ratings.fillna(0)

# Users bich cosine similarity calculate garam
similarity_matrix = cosine_similarity(centered_ratings_filled)

user_similarity = pd.DataFrame(
    similarity_matrix,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

# Recommendation dine user select garam
user_id = 1

# Aafailai hataera positive similarity bhayeka users chhanam
similar_users = (
    user_similarity[user_id]
    .drop(user_id)
    .loc[lambda scores: scores > 0]
    .sort_values(ascending=False)
)

# Top 10 similar users chhanam
top_users = similar_users.head(10)

print(f"Top 10 users similar to User {user_id}:")
print(top_users.to_string())

# Similar users ko ratings ra rating differences nikalam
neighbor_ratings = ratings_matrix.loc[top_users.index]
neighbor_deviations = centered_ratings.loc[top_users.index]

# Similarity weights lai table ma apply garam
weights = top_users.to_numpy().reshape(-1, 1)

# Missing ratings ko mask banam
rated_mask = neighbor_ratings.notna()

# Missing ratings lai calculation ma include nagaram
weighted_deviations = (
    neighbor_deviations.fillna(0)
    .mul(top_users, axis=0)
    .sum(axis=0)
)

# Movie rate gareka users ko positive similarity weight jodam
available_weights = (
    rated_mask.mul(top_users, axis=0)
    .sum(axis=0)
)

# Movie rate gareka similar users ko count nikalam
neighbor_counts = rated_mask.sum(axis=0)

# Weighted average deviation calculate garam
weighted_average_deviation = (
    weighted_deviations /
    available_weights.replace(0, float('nan'))
)

# User ko average ma deviation add garera prediction banam
predicted_ratings = (
    user_means.loc[user_id] + weighted_average_deviation
)

# User le pahile rate nagareka movies matra rakhau
unrated_mask = ratings_matrix.loc[user_id].isna()
predicted_ratings = predicted_ratings[unrated_mask]

# Kamti ma 2 jana similar users le rate gareka movies rakhau
valid_movies = neighbor_counts.loc[predicted_ratings.index] >= 2
predicted_ratings = predicted_ratings[valid_movies]

# Prediction ko range heram
print("\nPredicted rating statistics before clipping:")
print(predicted_ratings.describe())

# Rating lai valid range bhitra rakhau
predicted_ratings = predicted_ratings.clip(0.5, 5.0)

# Highest predicted ratings bhayeka top 10 movies chhanam
predicted_ratings = predicted_ratings.sort_values(
    ascending=False
).head(10)

# Movie ko title ra genres jodam
recommendations = movies[
    movies['movieId'].isin(predicted_ratings.index)
].copy()

recommendations['predicted_rating'] = (
    recommendations['movieId'].map(predicted_ratings)
)

recommendations = recommendations.sort_values(
    'predicted_rating',
    ascending=False
)

# Recommended movies print garam
print(f"\nTop recommended movies for User {user_id}:")

if recommendations.empty:
    print("No recommendations found with the current criteria.")
else:
    print(
        recommendations[
            ['title', 'genres', 'predicted_rating']
        ].to_string(index=False)
    )
