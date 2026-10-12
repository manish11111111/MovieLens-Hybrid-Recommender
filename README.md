# MovieLens Hybrid Movie Recommendation System

## Overview

This project explores movie recommendation techniques using the MovieLens dataset. It compares popularity-based recommendation, content-based filtering, collaborative filtering, and a hybrid approach that combines collaborative and content-based scores.

The main objective is to investigate whether combining user-rating patterns with movie genre information can improve recommendation ranking performance.

## Recommendation Approaches

- **Popularity-Based Recommendation:** Ranks movies using rating counts and average ratings.
- **Content-Based Filtering:** Uses TF-IDF and cosine similarity to identify movies with similar genres.
- **Collaborative Filtering:** Uses similarities between users and their rating patterns to estimate preferences.
- **Hybrid Recommendation:** Combines collaborative-filtering and genre-based content scores.

## Dataset

This project uses the [MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/latest/), provided by GroupLens Research.

The dataset files are not included in this repository to keep it lightweight.

### Setup

1. Download the `ml-latest-small.zip` dataset from the [official MovieLens website](https://grouplens.org/datasets/movielens/latest/).
2. Extract the ZIP file.
3. Copy the following files into the project's `data/` directory:
   - `movies.csv`
   - `ratings.csv`
   - `tags.csv`

Your project structure should look like this:

```text
MovieLens-Project/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── src/
├── README.md
├── .gitignore
└── test_results.png
```

After placing the files in the correct directory, run the relevant Python script from the project root.

## Evaluation Methodology

The dataset was divided into training, validation, and test sets.

| Split | Number of ratings |
|---|---:|
| Training | 70,585 |
| Validation | 10,083 |
| Test | 20,168 |

Hybrid model weights were compared on the validation set. The 50% collaborative-filtering and 50% content-based configuration achieved the best validation F1@10 among the tested configurations.

The selected model was then evaluated on the held-out test set.

## Final Test Results

| Metric | Collaborative Filtering | Hybrid (50/50) |
|---|---:|---:|
| Precision@10 | 0.0146 | 0.0276 |
| Recall@10 | 0.0126 | 0.0246 |
| F1@10 | 0.0112 | 0.0210 |

In this experiment, the hybrid model outperformed the collaborative-filtering baseline on all three reported ranking metrics.

Compared with collaborative filtering, the hybrid model achieved approximately:
- 89% higher Precision@10
- 95% higher Recall@10
- 88% higher F1@10

These results come from one fixed random split. Additional experiments across multiple random seeds would help establish how consistent the improvements are.

## Evaluation Definitions

- **Precision@10:** The proportion of the top 10 recommendations that are relevant.
- **Recall@10:** The proportion of relevant held-out items that appear in the top 10 recommendations.
- **F1@10:** The harmonic mean of Precision@10 and Recall@10.

A held-out rating of 4.0 or higher is treated as relevant.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Cosine similarity
- VS Code

## Limitations

- Genre-based features provide limited information about a movie's actual content.
- The dataset is relatively small compared with production recommendation datasets.
- Results may vary across different data splits.
- Recommendation performance depends on the candidate-generation and scoring methods.

## Future Work

- Evaluate models across multiple random seeds.
- Compare against additional baseline recommenders.
- Investigate richer movie metadata.
- Develop an interactive interface for generating recommendations.
- Explore knowledge graphs to represent relationships between movies, actors, directors, and genres.

## How to Run
From the project root, activate the Python environment and run:

```powershell
python src/validate_hybrid.py
```
Make sure the required Python packages are installed and the MovieLens CSV files are available in the `data/` directory.