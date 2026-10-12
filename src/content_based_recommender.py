
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load movie and tag data
movies = pd.read_csv('data/movies.csv')
tags = pd.read_csv('data/tags.csv')

# Display the first 10 movies
print("Movie dataset:")
print(movies[['title', 'genres']].head(10).to_string(index=False))

# Step 3: Combine all tags belonging to each movie
tags_by_movie = (
    tags.groupby('movieId')['tag']
    .apply(lambda x: ' '.join(x.astype(str)))
    .reset_index()
)

# Merge tags with movie data
movies = movies.merge(
    tags_by_movie,
    on='movieId',
    how='left'
)

# Handle movies without tags
movies['tag'] = movies['tag'].fillna('')

# Combine genres and tags
movies['content'] = (
    movies['genres'].str.replace('|', ' ', regex=False)
    + ' '
    + movies['tag']
)

print("\nCombined movie content:")
print(movies[['title', 'content']].head(10).to_string(index=False))

# Convert content into TF-IDF features
tfidf = TfidfVectorizer(
    stop_words='english',
    token_pattern=r'(?u)\b\w+\b'
)

tfidf_matrix = tfidf.fit_transform(movies['content'])

print("\nTF-IDF matrix shape:", tfidf_matrix.shape)
print("Number of features:", len(tfidf.get_feature_names_out()))

# Calculate similarity between movies
similarity_matrix = cosine_similarity(tfidf_matrix)

print("Similarity matrix shape:", similarity_matrix.shape)

# Select a movie
movie_title = "Toy Story (1995)"

movie_matches = movies.index[
    movies['title'] == movie_title
]

if len(movie_matches) == 0:
    print(f"\nMovie not found: {movie_title}")

else:
    movie_index = movie_matches[0]

    # Get similarity scores
    similarity_scores = list(
        enumerate(similarity_matrix[movie_index])
    )

    # Sort by similarity score
    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Exclude the selected movie
    similar_movies = [
        (index, score)
        for index, score in similarity_scores
        if index != movie_index
    ][:10]

    # Display recommendations
    print(f"\nMovies similar to {movie_title}:")

    for index, score in similar_movies:
        print(
            f"{movies.iloc[index]['title']} "
            f"— similarity: {score:.3f}"
        )
