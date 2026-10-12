
import pandas as pd

# Let's load movie and ratings data
movies = pd.read_csv('data/movies.csv')
ratings = pd.read_csv('data/ratings.csv')

# Calculate rating count and average rating for each movie
movie_stats = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    average_rating=('rating', 'mean')
).reset_index()

# Add movie titles
movie_stats = movie_stats.merge(
    movies[['movieId', 'title']],
    on='movieId',
    how='left'
)

# Keep movies with at least 50 ratings
popular_movies = movie_stats[
    movie_stats['rating_count'] >= 50
].copy()

# Sort by average rating, highest first
popular_movies = popular_movies.sort_values(
    'average_rating',
    ascending=False
)

# Display the top 10 popular movies
print("Top 10 Popular Movies:")
print(
    popular_movies[
        ['title', 'average_rating', 'rating_count']
    ].head(10).to_string(index=False)
)



# Calculate the overall average rating
C = ratings['rating'].mean()

# Minimum number of ratings required
m = 50

# Keep movies with at least 50 ratings
popular_movies = movie_stats[
    movie_stats['rating_count'] >= m
].copy()

# Calculate weighted rating
popular_movies['weighted_rating'] = (
    (popular_movies['rating_count'] /
     (popular_movies['rating_count'] + m)
     * popular_movies['average_rating'])
    +
    (m /
     (popular_movies['rating_count'] + m)
     * C)
)

# Rank movies by weighted rating
popular_movies = popular_movies.sort_values(
    'weighted_rating',
    ascending=False
)

# Display the top 10
print("\nTop 10 Movies by Weighted Rating:")
print(
    popular_movies[
        ['title', 'average_rating',
         'rating_count', 'weighted_rating']
    ].head(10).to_string(index=False)
)
