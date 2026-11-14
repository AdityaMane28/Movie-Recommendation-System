# 🎬 Movie Recommendation System

An end-to-end machine learning pipeline that recommends movies using **content-based filtering**, **collaborative filtering** and a **hybrid** of both. It is built on the MovieLens data format (`movies.csv` + `ratings.csv`) with Python and Scikit-learn.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Output](#example-output)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

With thousands of titles on streaming platforms, users struggle to find something worth watching. This project builds a recommender that learns from each user's rating history to suggest movies they are likely to enjoy, and finds movies similar to any title using rating patterns and genre information.

## Features

- 🎯 **Personalized recommendations** from each user's past ratings
- 🔍 **Similar-movie search** blending rating-matrix cosine similarity and genre/tag TF-IDF cosine similarity
- 🧠 **Content-based filtering** using TF-IDF on genres, decade and (optionally) tags or plot summaries
- 🤝 **Collaborative filtering** with item-based kNN and SVD matrix factorization
- 🔀 **Hybrid model** that blends collaborative and content-based scores
- ❄️ **Cold-start support**: recommendations for a new user from a few liked titles
- 💡 **Explanations**: "because you liked *X*" for item-based recommendations
- 📊 **Full evaluation**: RMSE and MAE for rating prediction; Precision, Recall, NDCG, HitRate and Coverage at K for ranking

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.9+ |
| Data | NumPy, Pandas, SciPy |
| Machine Learning | Scikit-learn (TF-IDF, cosine similarity), SciPy SVD |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

## Project Structure

```
Movie-Recommendation-System/
│
├── Movie_Recommender_Pipeline.ipynb   # Step-by-step notebook (one cell per stage)
├── movie_recommender.py               # Same pipeline as a command-line script
├── movies.csv                         # Dataset (add yourself)
├── ratings.csv                        # Dataset (add yourself)
├── requirements.txt                   # Dependencies
├── README.md
└── outputs/                           # Created on run: plots, metrics, saved models
```

## Dataset

The project uses the **MovieLens** file format. The [MovieLens datasets](https://grouplens.org/datasets/movielens/) are free for research use from GroupLens (check their license terms before redistributing).

| File | Columns |
|------|---------|
| `movies.csv` | `movieId`, `title`, `genres` (separated by `\|`) |
| `ratings.csv` | `userId`, `movieId`, `rating`, `timestamp` |
| `tags.csv` *(optional)* | `userId`, `movieId`, `tag`, `timestamp` |

> MovieLens has no plot summaries. To use plot text, add an `overview` column to `movies.csv` (for example from TMDB) or provide `tags.csv`.
> Don't re-save the CSVs from Excel: it truncates the timestamp column.

If the CSV files are missing, the notebook generates a small synthetic demo dataset so you can still run it.

## How It Works

1. **Load & explore**: dataset summary, rating distribution, long-tail and genre charts
2. **Preprocess**: extract release year, build content text (genres + decade + tags), drop users/movies with fewer than 5 ratings
3. **Split**: hold out 20% of *each user's* ratings for testing
4. **Train models**
   - **Popularity**: Bayesian weighted rating (non-personalized baseline)
   - **Bias baseline**: global mean + user bias + item bias
   - **Content-based**: TF-IDF vectors, with a taste profile built from the user's ratings
   - **Item-kNN CF**: cosine similarity on the user-rating matrix, with shrinkage
   - **SVD CF**: truncated SVD on the bias-removed rating matrix
   - **Hybrid**: `α · z(CF score) + (1 − α) · z(content score)`
5. **Evaluate**: error metrics and top-N ranking metrics on held-out data
6. **Recommend**: personalized lists, similar movies, cold-start users
7. **Save**: trained models and metrics go to `outputs/`

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

**Option A – Notebook (recommended)**
```bash
jupyter notebook
```
Open `Movie_Recommender_Pipeline.ipynb` and choose **Kernel → Restart & Run All**.

**Option B – Command line**
```bash
python movie_recommender.py --movies movies.csv --ratings ratings.csv \
       --user-id 1 --movie "Toy Story" --liked "Toy Story" "Jumanji"
```

Useful options: `--tags tags.csv`, `--svd-factors 20`, `--alpha 0.7` (hybrid weight on collaborative filtering), `--top-k 10`, `--min-movie-ratings 5`.

**Use the API in your own code**
```python
recommend_for_user(hybrid, ds, user_id=1, n=10)              # personalized top-N
similar_movies(ds, content, knn, "Toy Story", n=10)          # similar titles
recommend_for_new_user(ds, content, ["Toy Story"], n=10)     # cold start
```

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| Rating prediction | RMSE, MAE (lower is better) |
| Top-N ranking | Precision@K, Recall@K, NDCG@K, HitRate@K, Coverage (higher is better) |

A held-out rating of 4.0 or higher counts as a movie the user liked.

> 📌 Run the notebook on your data and paste your results table here.

| Model | RMSE | MAE | NDCG@10 |
|-------|------|-----|---------|
| Global mean | – | – | – |
| Item-kNN CF | – | – | – |
| SVD CF | – | – | – |
| Hybrid | – | – | – |

## Example Output

```
Movies similar to: Toy Story (1995)
        title        genres        content_sim   rating_sim
        ...          ...           ...           ...
```

*(Replace with your own output after running the notebook.)*

## Limitations

- Content-based filtering uses genres and decade only unless you add tags or plot text
- The item-similarity matrix is dense, so very large datasets (for example MovieLens 25M) need a higher `--min-movie-ratings` to fit in memory
- Cold-start users need at least a few liked titles to get personalized results

## Future Improvements

- Add sentence-embedding features from plot summaries
- Use implicit-feedback models such as ALS or LightFM
- Tune hyper-parameters (SVD factors, neighbours, hybrid weight) with cross-validation
- Use time-aware splits based on the rating timestamp
- Serve recommendations through a Streamlit or Flask app

## Contributing

Contributions are welcome. Fork the repo, create a feature branch and open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for details. The MovieLens data has its own license from GroupLens.

---

⭐ If you found this project useful, consider giving it a star.
