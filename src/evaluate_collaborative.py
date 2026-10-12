
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv('data/ratings.csv')

# Ratings lai training ra testing ma divide garam
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

print("Training ratings:", len(train_data))
print("Testing ratings:", len(test_data))

# Training data bata user ra movie ko table banam
ratings_matrix = train_data.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# Training data ko global ra user average nikalam
global_mean = train_data['rating'].mean()
user_means = ratings_matrix.mean(axis=1)

# User ko average bata rating ko difference nikalam
centered_ratings = ratings_matrix.sub(user_means, axis=0)

# Rating nadeko thau lai 0 banam
centered_ratings = centered_ratings.fillna(0)

# Training data bata user similarity calculate garam
similarity_matrix = cosine_similarity(centered_ratings)

user_similarity = pd.DataFrame(
    similarity_matrix,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

# Test ratings ko predictions store garam
actual_ratings = []
predicted_ratings = []

# Test ko pratyek rating ko prediction calculate garam
for row in test_data.itertuples(index=False):
    user_id = row.userId
    movie_id = row.movieId
    actual_rating = row.rating

    # User ra movie training data ma chha ki heram
    if (
        user_id not in ratings_matrix.index
        or movie_id not in ratings_matrix.columns
    ):
        prediction = global_mean

    else:
        # User ko average lai baseline banam
        user_mean = user_means.loc[user_id]

        # Yo movie rate gareka users nikalam
        movie_ratings = ratings_matrix[movie_id].dropna()
        neighbor_ids = movie_ratings.index

        # Target user sanga similarity nikalam
        similarities = user_similarity.loc[
            user_id, neighbor_ids
        ].drop(index=user_id, errors='ignore')

        # Positive similarity bhayeka users matra rakhau
        similarities = similarities[similarities > 0]

        # Sabai bhanda similar 20 users chhanam
        similarities = similarities.sort_values(
            ascending=False
        ).head(20)

        # Similar users available chhan ki heram
        if similarities.empty:
            prediction = user_mean

        else:
            # Similar users ko rating ra average nikalam
            neighbor_ids = similarities.index
            neighbor_movie_ratings = ratings_matrix.loc[
                neighbor_ids, movie_id
            ]
            neighbor_average = user_means.loc[neighbor_ids]

            # Rating average bata kati farak chha calculate garam
            deviations = neighbor_movie_ratings - neighbor_average

            # Similarity anusar weighted deviation calculate garam
            weighted_deviation = (
                deviations * similarities
            ).sum() / similarities.sum()

            # User ko average ma deviation add garam
            prediction = user_mean + weighted_deviation

    # Prediction lai valid rating range ma rakhau
    prediction = np.clip(prediction, 0.5, 5.0)

    actual_ratings.append(actual_rating)
    predicted_ratings.append(prediction)

# Model ko evaluation metrics calculate garam
rmse = np.sqrt(
    mean_squared_error(actual_ratings, predicted_ratings)
)

mae = mean_absolute_error(
    actual_ratings, predicted_ratings
)

# Evaluation result print garam
print("\nCollaborative Filtering Evaluation")
print("----------------------------------")
print(f"Global average rating: {global_mean:.3f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")

# Actual ra predicted rating compare garam
results = pd.DataFrame({
    'actual_rating': actual_ratings,
    'predicted_rating': predicted_ratings
})

print("\nSample predictions:")
print(results.head(10).to_string(index=False))
