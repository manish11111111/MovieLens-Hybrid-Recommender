# MovieLens Hybrid Movie Recommendation System

## Overview

This project is about building a movie recommendation system using the MovieLens dataset. It explores different ways to recommend movies, including popularity-based recommendations, content-based filtering, collaborative filtering, and a hybrid approach.

The main goal is to see whether combining information about movies with users' rating patterns can produce better recommendations.

The project also includes a Streamlit web application where users can select a MovieLens user, choose their favourite movies, and compare recommendations from collaborative filtering and the hybrid model.

## Recommendation Methods

- **Popularity-Based Recommendation:** Recommends movies based on their average ratings and the number of ratings they have received.
- **Content-Based Filtering:** Uses TF-IDF and cosine similarity to find movies with similar genres and tags.
- **Collaborative Filtering:** Looks at the ratings of similar users to predict which movies a user might enjoy.
- **Hybrid Recommendation:** Combines collaborative-filtering predictions with content-based similarity scores to rank movies.

## Dataset

This project uses the [MovieLens latest-small dataset](https://grouplens.org/datasets/movielens/latest/) from GroupLens Research.

The dataset is not included in this repository, so it needs to be downloaded separately.

### Getting Started

1. Download `ml-latest-small.zip` from the [official MovieLens website](https://grouplens.org/datasets/movielens/latest/).
2. Extract the downloaded ZIP file.
3. Copy these three files into the project's `data/` directory:
   - `movies.csv`
   - `ratings.csv`
   - `tags.csv`

The project structure should look similar to this:

```text
MovieLens-Project/
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│   └── tags.csv
├── src/
│   └── validate_hybrid.py
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── test_results.png
```

## Model Evaluation

To compare the recommendation methods, the ratings data was divided into training, validation, and test sets.

| Dataset Split | Number of Ratings |
|---|---:|
| Training | 70,585 |
| Validation | 10,083 |
| Test | 20,168 |

Different combinations of collaborative-filtering and content-based weights were tested on the validation set. Among the configurations tested, the 50/50 combination achieved the best F1@10 score.

The selected configuration was then evaluated using the held-out test set.

## Test Results

| Metric | Collaborative Filtering | Hybrid (50/50) |
|---|---:|---:|
| Precision@10 | 0.0146 | 0.0276 |
| Recall@10 | 0.0126 | 0.0246 |
| F1@10 | 0.0112 | 0.0210 |

In this experiment, the hybrid model performed better than the collaborative-filtering baseline across all three metrics.

The hybrid model achieved approximately:

- **89% higher Precision@10**
- **95% higher Recall@10**
- **88% higher F1@10**

These results suggest that combining rating patterns with movie information can improve recommendation ranking. However, the evaluation used a single random split, so further testing with different random seeds would help determine whether similar improvements can be achieved consistently.

## Interactive Web Application

The project includes a Streamlit application for exploring movie recommendations.

Users can:

- Select a MovieLens user ID.
- Choose their favourite movies.
- Generate collaborative-filtering recommendations.
- Compare them with hybrid recommendations.
- View predicted ratings and recommendation scores.
- Download the results as CSV files.

The current application uses a **70% collaborative-filtering and 30% content-based** weighting for its hybrid recommendations.

This differs from the 50/50 configuration used for the evaluation results above. The reported test metrics therefore describe the tested 50/50 model, not necessarily the current application configuration.

Movies without valid collaborative-filtering predictions are excluded rather than assigned a fallback rating. The current implementation also requires at least two similar users to have rated a candidate movie.

## Evaluation Metrics

- **Precision@10:** Measures how many of the top 10 recommended movies are relevant.
- **Recall@10:** Measures how many relevant movies appear among the top 10 recommendations.
- **F1@10:** Combines precision and recall into a single metric.

For this project, a movie with a held-out rating of 4.0 or higher is considered relevant.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- TF-IDF
- Cosine similarity
- Streamlit
- VS Code

## Limitations

- Genres and tags do not capture every aspect of a movie.
- The dataset is relatively small compared with those used by large commercial recommendation platforms.
- Results can change depending on how the data is split.
- Recommendations depend on the available ratings and the way candidate movies are selected and ranked.
- Users or movies with limited rating data can be difficult to handle accurately.

## Future Improvements

There are several ways this project could be improved:

- Test the models using multiple random seeds.
- Compare the results with additional recommendation methods.
- Include more detailed movie information.
- Improve the ranking and evaluation process.
- Explore knowledge graphs to represent connections between movies, actors, directors, and genres.
- Make the recommendation application available online.

## How to Run

### 1. Install the Required Packages

Open a terminal in the project directory and run:

```powershell
pip install -r requirements.txt
```

### 2. Run the Evaluation Script

Make sure the MovieLens dataset is available in the `data/` directory, then run:

```powershell
python src/validate_hybrid.py
```

### 3. Start the Web Application

Run the following command from the project root:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal to use the application.

## Acknowledgements

Thanks to [GroupLens Research](https://grouplens.org/) for providing the MovieLens dataset used in this project.