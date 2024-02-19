# Movie Recommendation System

## Introduction
This project is a movie recommendation system that utilizes content-based filtering and machine learning algorithms to suggest movies to users based on their preferences and previous ratings. It employs natural language processing (NLP) techniques to analyze movie titles and genres, and it utilizes various machine learning models for classification and recommendation.

## Dependencies
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - nltk
  - scikit-learn
  - matplotlib
  - seaborn

## Installation
1. Clone the repository: `git clone https://github.com/your_username/your_repository.git`
2. Navigate to the project directory: `cd your_repository`

## Usage
1. Make sure you have installed all the dependencies mentioned above.
2. Run the main script: `python main.py`
3. Follow the prompts or execute specific functions as needed.

## Data
The project uses three main datasets:
- `movies.csv`: Contains information about movies, including titles and genres.
- `users.csv`: Contains information about users, including demographic data.
- `ratings.csv`: Contains user ratings for different movies.

## Preprocessing
- The datasets are loaded using pandas, and initial exploration is performed to check for duplicate rows and missing values.
- Text preprocessing techniques are applied to clean and normalize the movie titles and genres, including removing numbers, punctuation, and stopwords, as well as tokenization, stemming, and lemmatization.

## Feature Engineering
- The datasets are transformed and scaled as necessary for modeling purposes, including normalizing zip codes and timestamps.
- Gender information is encoded using one-hot encoding.

## Modeling
- The project implements several machine learning models, including Naive Bayes, SVM, and Decision Trees, for movie classification and recommendation.
- The models are trained and evaluated using accuracy metrics on a subset of the data.


## Functionality
- The project provides two main recommendation functions:
  1. `Recommender_using_countvictorizer(movie)`: Recommends movies based on count vectorization similarity.
  2. `Recommender_using_Tfidf(movie)`: Recommends movies based on TF-IDF similarity.


