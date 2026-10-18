# MovieLens Hybrid Recommender System

A movie recommendation system that combines collaborative filtering and content-based filtering to generate personalized movie recommendations.

**Live Demo:** https://movielens-hybrid-recommender1111.streamlit.app/

**Repository:** https://github.com/manish11111111/MovieLens-Hybrid-Recommender

## About the Project

I built this project to explore how recommendation systems work and how machine learning techniques can help personalize movie suggestions.

The system combines two approaches. Collaborative filtering uses patterns in user ratings, while content-based filtering uses movie characteristics such as genres and tags.

I evaluated different combinations of these approaches and selected a 50/50 hybrid configuration based on validation results.

## Features

- Personalized movie recommendations
- User selection and favorite movie selection
- Adjustable number of recommendations
- Interactive interface built with Streamlit
- Downloadable recommendations in CSV format
- Model evaluation using Precision@10, Recall@10 and F1@10

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Git and GitHub

## How It Works

### Collaborative Filtering

Uses patterns in user ratings to estimate which movies a user may enjoy.

### Content-Based Filtering

Uses movie information, including genres and tags, to identify movies with similar characteristics.

### Hybrid Recommendation

Combines both approaches using equal weights:

`Hybrid Score = 0.5 × CF Score + 0.5 × Content Score`

The 50/50 configuration achieved the highest F1@10 among the configurations evaluated on the validation set.

## Model Evaluation

The dataset was divided into training, validation and test sets using a 70/10/20 split.

| Dataset split | Number of ratings | Percentage |
|---|---:|---:|
| Training | 70,585 | 70% |
| Validation | 10,083 | 10% |
| Test | 20,168 | 20% |

### Final Test Results

| Model | Precision@10 | Recall@10 | F1@10 |
|---|---:|---:|---:|
| Collaborative Filtering | 0.0146 | 0.0126 | 0.0112 |
| Hybrid (50/50) | 0.0276 | 0.0246 | 0.0210 |

The hybrid model achieved an F1@10 of 0.0210, compared with 0.0112 for the collaborative filtering baseline. This represents an approximately 87.5% relative improvement in F1@10 on the test set.

These results indicate that combining rating patterns with movie content improved performance in this evaluation.

**Evaluation note:** The validation script uses genres for content features, while the deployed application also uses tags. As a result, recommendations from the live application may differ from the evaluation results.

## Running the Project Locally

### Prerequisites

- Python 3.10 or later
- Git
- The required MovieLens dataset files

### 1. Clone the repository

```bash
git clone https://github.com/manish11111111/MovieLens-Hybrid-Recommender.git
cd MovieLens-Hybrid-Recommender
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

On Windows, activate it using:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Check the dataset

Ensure the required CSV files are available in the `data` directory:

```text
data/
├── ratings.csv
├── movies.csv
└── tags.csv
```

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit in your browser.

## Project Structure

```text
MovieLens-Hybrid-Recommender/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   └── tags.csv
└── src/
    └── validate_hybrid.py
```

## What I Learned

This project gave me practical experience with data preparation, feature extraction, similarity-based recommendation techniques, model evaluation and deploying a machine learning application.

It also helped me understand the importance of selecting model configurations using validation data and evaluating final performance on a separate test set.

## Future Improvements

- Improve recommendation quality and cold-start handling
- Explore additional recommendation algorithms
- Add automated tests and improve reproducibility
- Explore database integration and scalable data pipelines

## Author

**Manish Chapagain**

Computer Engineering Graduate | Aspiring Data & AI Engineer

This project is part of my ongoing journey into data science, machine learning and AI engineering.
