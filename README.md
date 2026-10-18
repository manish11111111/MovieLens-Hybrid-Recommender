# MovieLens – AI Movie Recommender

A movie recommendation system built with Python, Streamlit, and machine learning. It combines collaborative filtering with content-based filtering to recommend movies based on a user's preferences and the behavior of similar users.

## Features

- **Hybrid recommendations:** Combines collaborative filtering and content-based filtering with equal weights.
- **Personalized suggestions:** Uses a selected user's rating history and favorite movies.
- **Movie discovery:** Displays recommended movies with their scores and available movie information.
- **Adjustable results:** Choose how many recommendations to display.
- **CSV export:** Download recommendation results for further analysis.

## How It Works

The recommender combines two approaches:

- **Collaborative filtering (50%):** Uses rating patterns and similarities between users to estimate which movies a user may enjoy.
- **Content-based filtering (50%):** Uses movie features, including genres and tags in the interactive application, to find movies related to a user's preferences.

The final hybrid score is calculated as:

`Hybrid Score = 0.5 × CF Score + 0.5 × Content Score`

The scores are normalized before combining them so that both approaches contribute on comparable scales.

## Dataset

This project uses the MovieLens latest-small dataset, which contains movie information and user ratings.

The ratings dataset was divided into three subsets:

| Split | Ratings | Percentage |
|---|---:|---:|
| Training | 70,585 | 70% |
| Validation | 10,083 | 10% |
| Test | 20,168 | 20% |

The validation set was used to compare hybrid configurations. The selected configuration was then evaluated on the held-out test set.

## Model Evaluation

The validation experiment compared five combinations of collaborative filtering and content-based filtering. Hybrid 50/50 achieved the highest validation F1@10 and was selected for final testing.

### Final test results

| Metric | Collaborative Filtering | Hybrid 50/50 |
|---|---:|---:|
| Precision@10 | 0.0146 | 0.0276 |
| Recall@10 | 0.0126 | 0.0246 |
| F1@10 | 0.0112 | 0.0210 |

The hybrid model outperformed the collaborative filtering baseline on all three reported ranking metrics in this experiment.

These results represent offline evaluation on the specified data split. They do not guarantee that every recommendation will match an individual user's preferences.

### Evaluation and application differences

The validation script uses movie genres with TF-IDF for content-based features. The interactive application also incorporates tags. Therefore, the application uses the same 50/50 weighting strategy but its recommendations may differ from those produced by the evaluation script.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- TF-IDF vectorization
- Cosine similarity

## Project Structure

```text
MovieLens-Project/
├── app.py
├── requirements.txt
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
└── src/
    └── validate_hybrid.py
```

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd MovieLens-Project
```

Replace `YOUR_REPOSITORY_URL` with your GitHub repository URL.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Streamlit will provide a local URL where you can open the application in your browser.

## Limitations

- Recommendations depend on the quality and availability of movie ratings and metadata.
- New users with little or no rating history may receive less personalized recommendations.
- The offline evaluation uses a particular random data split, so results may vary with other splits or evaluation methods.
- The current evaluation and interactive application use different content features.

## Future Improvements

- Improve recommendations for new users and movies.
- Compare additional ranking metrics and validation strategies.
- Experiment with more advanced recommendation algorithms.
- Add movie posters and richer movie metadata.
- Evaluate recommendations using user feedback.

## Author

Developed as a machine learning and recommendation systems project using the MovieLens dataset.
