import pandas as pd

#lets load movie and ratings data
movies= pd.read_csv('data/movies.csv')
ratings= pd.read_csv('data/ratings.csv')

#lets display those data
print('Movies Dataset:')
print(movies.head().to_string(index=False))

print('\nRatings Dataset:')
print(ratings.head().to_string(index=False))

print("\nTotal movies:",len(movies))
print("Total Ratings:",len(ratings))
print("Unique Users:",ratings['userId'].nunique())

#Data lai analyze garam(rows,colums,datatype k xa vanera heram)
print("\nMovies Dataset Info:")
print("shape:",movies.shape)
print("columns:",movies.columns)
print("data types:\n",movies.dtypes)
print(movies.info())

print("\nRatings Dataset Info:")
print("shape:",ratings.shape)
print("columns:",ratings.columns)
print("data types:\n",ratings.dtypes)
print(ratings.info())

#MIssing values check garam
print("\nMissing Values in Movies Dataset:")
print(movies.isnull().sum())
print("\nMissing Values in Ratings Dataset:")   
print(ratings.isnull().sum())

#Duplicated values check garam
print("\nDuplicated Values in Movies Dataset:")
print(movies.duplicated().sum())    
print("\nDuplicated Values in Ratings Dataset:")
print(ratings.duplicated().sum())

#Aba Ratings alalyze garam
print("\nRatings Statistics:")
print(ratings["rating"].value_counts().sort_index())
print("\nAverage Rating:", ratings["rating"].mean())
print("Median Rating:", ratings["rating"].median())
print("Lowest Rating:", ratings["rating"].min())
print("Highest Rating:", ratings["rating"].max())

print("\nMost Rated Movies:")
print(ratings["movieId"].value_counts().head(10))


#Aba most rated movies ko details heram
print("\nDetails of Most Rated Movies:")
most_rated_movies = ratings["movieId"].value_counts().head(10).index
#Index bina print garna ko lagi isin() function use garam
print(movies[movies["movieId"].isin(most_rated_movies)]
    .to_string(index=False)
)
