
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Data load garam
ratings = pd.read_csv('data/ratings.csv')
movies = pd.read_csv('data/movies.csv')
tags = pd.read_csv('data/tags.csv')

# Training ra testing data banam
train_data, test_data = train_test_split(
    ratings,
    test_size=0.2,
    random_state=42
)

# Movie ko tags jodam
tags_by_movie = (
    tags.groupby('movieId')['tag']
    .apply(lambda values: ' '.join(values.astype(str)))
    .reset_index()
)

# Tags lai movies sanga jodam
movies = movies.merge(
    tags_by_movie,
    on='movieId',
    how='left'
)

movies['tag'] = movies['tag'].fillna('')

# Genre ra tags combine garam
movies['content'] = (
    movies['genres'].str.replace('|', ' ', regex=False)
    + ' '
    + movies['tag']
)

# Text lai TF-IDF features ma convert garam
tfidf = TfidfVectorizer(
    stop_words='english',
    token_pattern=r'(?u)\b\w+\b'
)

tfidf_matrix = tfidf.fit_transform(movies['content'])

# Movies bich similarity calculate garam
similarity_matrix = cosine_similarity(tfidf_matrix)

# Movie ID bata matrix index khojna mapping banam
movie_to_index = pd.Series(
    movies.index,
    index=movies['movieId']
).to_dict()

# User ko test ratings bata relevant movies banam
relevant_test = test_data[test_data['rating'] >= 4.0]

relevant_by_user = (
    relevant_test.groupby('userId')['movieId']
    .apply(set)
    .to_dict()
)

# Training data bata user ko liked movies nikalam
liked_train = train_data[train_data['rating'] >= 4.0]

liked_by_user = (
    liked_train.groupby('userId')['movieId']
    .apply(list)
    .to_dict()
)

# Evaluation ko result store garam
precision_scores = []
recall_scores = []
evaluated_users = 0

# Pratyek user ko recommendations calculate garam
for user_id, relevant_movies in relevant_by_user.items():

    # User ko training ma liked movies chhan ki heram
    liked_movies = liked_by_user.get(user_id, [])

    if not liked_movies:
        continue

    # User le pahile rate gareka movies hatauna set banam
    seen_movies = set(
        train_data.loc[
            train_data['userId'] == user_id,
            'movieId'
        ]
    )

    # Similarity scores jodna array banam
    total_scores = np.zeros(len(movies))

    # User ko harek liked movie bata similar movies khojam
    for movie_id in liked_movies:
        movie_index = movie_to_index.get(movie_id)

        if movie_index is not None:
            total_scores += similarity_matrix[movie_index]

    # Dherai liked movies bhaye average similarity nikalam
    total_scores = total_scores / len(liked_movies)

    # User le pahile rate gareka movies exclude garam
    candidate_indices = [
        movie_to_index[movie_id]
        for movie_id in movie_to_index
        if movie_id not in seen_movies
    ]

    # Candidate movies lai similarity anusar rank garam
    ranked_indices = sorted(
        candidate_indices,
        key=lambda index: total_scores[index],
        reverse=True
    )

    # Top 10 movie IDs nikalam
    top_10_movies = set(
        movies.iloc[ranked_indices[:10]]['movieId']
    )

    # Relevant recommendations count garam
    hits = len(top_10_movies.intersection(relevant_movies))

    # Precision ra recall calculate garam
    precision_scores.append(hits / 10)
    recall_scores.append(hits / len(relevant_movies))
    evaluated_users += 1

# Average result print garam
print("\nContent-Based Ranking Results")
print("-----------------------------")
print(f"Users evaluated: {evaluated_users}")

if evaluated_users > 0:
    print(f"Precision@10: {np.mean(precision_scores):.4f}")
    print(f"Recall@10: {np.mean(recall_scores):.4f}")
else:
    print("No users could be evaluated.")
