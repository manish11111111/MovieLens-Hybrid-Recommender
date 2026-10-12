
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv('data/ratings.csv')

# Training ra testing data banam
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Training data bata average ratings nikalam
global_mean = train_data['rating'].mean()

user_means = train_data.groupby('userId')['rating'].mean()
movie_means = train_data.groupby('movieId')['rating'].mean()

# User ra movie ko rating matrix banam
ratings_matrix = train_data.pivot_table(
    index='userId',
    columns='movieId',
    values='rating'
)

# User ko average bata rating ko difference nikalam
centered_ratings = ratings_matrix.sub(
    ratings_matrix.mean(axis=1),
    axis=0
).fillna(0)

# Users bich similarity calculate garam
similarity_matrix = cosine_similarity(centered_ratings)

user_similarity = pd.DataFrame(
    similarity_matrix,
    index=ratings_matrix.index,
    columns=ratings_matrix.index
)

# Sabai models ko prediction store garam
actual = []
global_predictions = []
user_predictions = []
movie_predictions = []
collaborative_predictions = []

# Test ko pratyek rating predict garam
for row in test_data.itertuples(index=False):
    user_id = row.userId
    movie_id = row.movieId
    actual_rating = row.rating

    # Global average bata prediction garam
    global_prediction = global_mean

    # User average bata prediction garam
    user_prediction = user_means.get(
        user_id, global_mean
    )

    # Movie average bata prediction garam
    movie_prediction = movie_means.get(
        movie_id, global_mean
    )

    # Collaborative prediction ko baseline user average ho
    collaborative_prediction = user_prediction

    # User ra movie training data ma chhan ki heram
    if (
        user_id in ratings_matrix.index
        and movie_id in ratings_matrix.columns
    ):
        # Yo movie rate gareka users nikalam
        movie_ratings = ratings_matrix[movie_id].dropna()

        # Target user sanga similarity nikalam
        similarities = user_similarity.loc[
            user_id, movie_ratings.index
        ].drop(index=user_id, errors='ignore')

        # Positive similarity bhayeka users matra rakhau
        similarities = similarities[similarities > 0]

        # Top 20 similar users chhanam
        similarities = similarities.sort_values(
            ascending=False
        ).head(20)

        # Similar users available chhan ki heram
        if not similarities.empty:
            neighbor_ids = similarities.index

            # Similar users ko rating ra average nikalam
            neighbor_ratings = ratings_matrix.loc[
                neighbor_ids, movie_id
            ]
            neighbor_averages = user_means.loc[neighbor_ids]

            # Average bata rating ko difference nikalam
            deviations = neighbor_ratings - neighbor_averages

            # Similarity anusar weighted difference calculate garam
            weighted_deviation = (
                deviations * similarities
            ).sum() / similarities.sum()

            # User average ma weighted difference add garam
            collaborative_prediction = (
                user_prediction + weighted_deviation
            )

    # Predictions lai valid rating range ma rakhau
    global_predictions.append(
        np.clip(global_prediction, 0.5, 5.0)
    )
    user_predictions.append(
        np.clip(user_prediction, 0.5, 5.0)
    )
    movie_predictions.append(
        np.clip(movie_prediction, 0.5, 5.0)
    )
    collaborative_predictions.append(
        np.clip(collaborative_prediction, 0.5, 5.0)
    )

    actual.append(actual_rating)

# Sabai models ko result compare garam
models = {
    'Global Average': global_predictions,
    'User Average': user_predictions,
    'Movie Average': movie_predictions,
    'Collaborative Filtering': collaborative_predictions
}

results = []

for model_name, predictions in models.items():
    rmse = np.sqrt(
        mean_squared_error(actual, predictions)
    )
    mae = mean_absolute_error(actual, predictions)

    results.append({
        'Model': model_name,
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4)
    })

# Result lai RMSE anusar sort garam
results_df = pd.DataFrame(results).sort_values('RMSE')

print("\nModel Comparison Results:")
print(results_df.to_string(index=False))
