
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class HybridRecommender:
    """MovieLens hybrid recommender using collaborative filtering and TF-IDF."""

    def __init__(self, top_neighbors=20, content_weight=0.30):
        self.top_neighbors = top_neighbors
        self.content_weight = content_weight
        self.cf_weight = 1.0 - content_weight

        self.movies = None
        self.ratings = None
        self.user_movie = None
        self.rating_array = None
        self.rating_mask = None
        self.user_means = None
        self.user_similarity = None
        self.user_ids = None
        self.user_to_index = None
        self.movie_ids = None
        self.movie_to_index = None
        self.rating_movie_ids = None
        self.movie_to_rating_col = None
        self.tfidf_matrix = None
        self.vectorizer = None

    def fit(self):
        """Load MovieLens data and prepare recommendation matrices."""
        ratings = pd.read_csv(DATA_DIR / "ratings.csv")
        movies = pd.read_csv(DATA_DIR / "movies.csv")
        tags = pd.read_csv(DATA_DIR / "tags.csv")

        tags_by_movie = (
            tags.groupby("movieId")["tag"]
            .apply(lambda values: " ".join(values.dropna().astype(str)))
            .reset_index()
        )

        movies = movies.merge(tags_by_movie, on="movieId", how="left")
        movies["tag"] = movies["tag"].fillna("")
        movies["genres"] = movies["genres"].fillna("")

        movies["content"] = (
            movies["genres"].str.replace("|", " ", regex=False)
            + " "
            + movies["tag"]
        )

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            movies["content"]
        )

        self.movies = movies.set_index("movieId", drop=False)
        self.ratings = ratings

        self.movie_ids = movies["movieId"].to_numpy()
        self.movie_to_index = {
            int(movie_id): index
            for index, movie_id in enumerate(self.movie_ids)
        }

        self.user_movie = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
        )

        self.rating_array = self.user_movie.to_numpy(dtype=np.float32)
        self.rating_mask = np.isfinite(self.rating_array)
        self.user_means = np.nanmean(self.rating_array, axis=1)

        centered = np.where(
            self.rating_mask,
            self.rating_array - self.user_means[:, None],
            0.0,
        )

        self.user_similarity = cosine_similarity(centered).astype(
            np.float32
        )
        np.fill_diagonal(self.user_similarity, 0.0)

        self.user_ids = self.user_movie.index.to_numpy()
        self.user_to_index = {
            int(user_id): index
            for index, user_id in enumerate(self.user_ids)
        }

        self.rating_movie_ids = self.user_movie.columns.to_numpy()
        self.movie_to_rating_col = {
            int(movie_id): index
            for index, movie_id in enumerate(self.rating_movie_ids)
        }

        return self

    def recommend(self, user_id=1, limit=10, favorite_ids=None):
        """
        Recommend unseen MovieLens movies.

        favorite_ids must contain MovieLens movie IDs, not TMDB IDs.
        If supplied, their content vectors form the user's preference profile.
        """
        user_id = int(user_id)
        limit = max(1, min(int(limit), 50))

        if user_id not in self.user_to_index:
            raise ValueError(f"Unknown MovieLens user ID {user_id}.")

        user_index = self.user_to_index[user_id]
        user_ratings = self.user_movie.loc[user_id].dropna()

        seen_movies = set(int(mid) for mid in user_ratings.index)

        if favorite_ids:
            valid_favorites = [
                int(mid)
                for mid in favorite_ids
                if int(mid) in self.movie_to_index
            ]
        else:
            valid_favorites = [
                int(mid)
                for mid, rating in user_ratings.items()
                if rating >= 4.0 and int(mid) in self.movie_to_index
            ]

        if not valid_favorites:
            raise ValueError(
                "No valid MovieLens favorite IDs were supplied."
            )

        liked_indices = [
            self.movie_to_index[mid] for mid in valid_favorites
        ]

        user_profile = self.tfidf_matrix[liked_indices].mean(axis=0)
        content_scores = cosine_similarity(
            np.asarray(user_profile),
            self.tfidf_matrix,
        ).ravel()

        similarities = self.user_similarity[user_index]
        neighbor_indices = np.argsort(similarities)[::-1][
            : self.top_neighbors
        ]
        neighbor_indices = [
            idx for idx in neighbor_indices if similarities[idx] > 0
        ]

        if neighbor_indices:
            neighbor_weights = similarities[neighbor_indices]
            neighbor_ratings = self.rating_array[neighbor_indices]
            neighbor_masks = self.rating_mask[neighbor_indices]

            deviations = np.where(
                neighbor_masks,
                neighbor_ratings - self.user_means[neighbor_indices, None],
                0.0,
            )

            weighted_sum = (
                neighbor_weights[:, None] * deviations
            ).sum(axis=0)

            weight_sum = (
                neighbor_weights[:, None] * neighbor_masks
            ).sum(axis=0)

            valid_columns = weight_sum > 0
            cf_predictions = np.full(
                len(self.rating_movie_ids),
                np.nan,
                dtype=np.float32,
            )

            cf_predictions[valid_columns] = (
                self.user_means[user_index]
                + weighted_sum[valid_columns] / weight_sum[valid_columns]
            )
        else:
            cf_predictions = np.full(
                len(self.rating_movie_ids),
                np.nan,
                dtype=np.float32,
            )

        # For a cold-start profile, use the overall mean rating as a
        # neutral fallback rather than failing when neighbors are absent.
        overall_mean = float(self.ratings["rating"].mean())

        candidates = [
            int(mid)
            for mid in self.movie_ids
            if int(mid) not in seen_movies
            and int(mid) not in valid_favorites
        ]

        if not candidates:
            return []

        candidate_indices = np.array(
            [self.movie_to_index[mid] for mid in candidates]
        )
        candidate_content = content_scores[candidate_indices]

        min_score = float(candidate_content.min())
        max_score = float(candidate_content.max())

        if max_score > min_score:
            candidate_content = (
                (candidate_content - min_score)
                / (max_score - min_score)
            )
        else:
            candidate_content = np.zeros(len(candidates))

        results = []

        for movie_id, content_score in zip(
            candidates, candidate_content
        ):
            rating_col = self.movie_to_rating_col.get(movie_id)

            if rating_col is not None and np.isfinite(
                cf_predictions[rating_col]
            ):
                predicted_rating = float(cf_predictions[rating_col])
            else:
                # No reliable neighbor prediction: use the dataset mean.
                predicted_rating = overall_mean

            cf_normalized = float(
                np.clip((predicted_rating - 0.5) / 4.5, 0.0, 1.0)
            )

            hybrid_score = (
                self.cf_weight * cf_normalized
                + self.content_weight * float(content_score)
            )

            movie = self.movies.loc[movie_id]

            results.append(
                {
                    "movieId": int(movie_id),
                    "title": str(movie["title"]),
                    "genres": str(movie["genres"]),
                    "predicted_rating": round(
                        float(np.clip(predicted_rating, 0.5, 5.0)), 2
                    ),
                    "score": round(float(hybrid_score), 4),
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )
        return results[:limit]
