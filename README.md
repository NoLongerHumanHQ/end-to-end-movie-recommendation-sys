# Movie Recommendation System

A complete end-to-end movie recommendation system built with Python and Streamlit.

## Features

- Multiple recommendation algorithms:
  - Content-Based Filtering: Recommends movies based on movie features (genres, keywords, etc.)
  - Collaborative Filtering: Recommends movies based on user behavior patterns
  - Matrix Factorization: Uses SVD for better recommendations
  - Hybrid Approach: Combines multiple algorithms for optimal results
- Interactive Streamlit web interface with:
  - Movie search functionality
  - Personalized recommendations
  - User profiles and ratings
  - Movie details with posters and metadata
- Integration with TMDB API for movie posters and additional data
- Comprehensive evaluation metrics

## Screenshot

![Movie Recommender System Screenshot](screenshots/app_screenshot.png)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file in the root directory (you can copy from `.env.example`)
   - Add your [TMDB API key](https://www.themoviedb.org/documentation/api) to the file:
     ```
     TMDB_API_KEY=your_key_here
     ```

4. Download the MovieLens dataset:
   ```
   python movie_recommendation_system/data/download_data.py
   ```
   - This script will let you choose between the small (100K ratings) or full (25M ratings) dataset
   - For testing, the small dataset is recommended

5. Process the data:
   ```
   python movie_recommendation_system/process_data.py
   ```
   - Add `--sample` flag for faster processing with a sample of the data
   - Add `--enrich` flag to fetch additional movie metadata from TMDB API

## Usage

Run the Streamlit app:

```
streamlit run movie_recommendation_system/app.py
```

This will start the web application on your local machine, typically at http://localhost:8501

### Using the Application

1. **Home Page**: Browse popular movies and get an overview of the system
2. **Movie Search**: Search for movies by title or genre
3. **Recommendations**: Get personalized recommendations based on movies you've rated
4. **User Profile**: View and manage your movie ratings
5. **About**: Learn more about the recommendation algorithms and project

## Project Structure

```
movie_recommendation_system/
├── app.py                    # Main Streamlit application
├── process_data.py           # Data preprocessing script
├── data/                     # Data directory
│   ├── download_data.py      # Script to download MovieLens dataset
│   ├── movies.csv            # Movie information
│   ├── ratings.csv           # User ratings
│   └── processed/            # Processed data files
├── models/                   # Recommendation algorithms
│   ├── content_based.py      # Content-based filtering
│   ├── collaborative.py      # Collaborative filtering
│   ├── hybrid.py             # Hybrid recommender
│   └── model_utils.py        # Utility functions for models
├── utils/                    # Utility modules
│   ├── data_processing.py    # Data processing utilities
│   ├── api_handlers.py       # TMDB API integration
│   └── recommendation_engine.py  # Main recommendation engine
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Data Processing

The system uses the MovieLens dataset, which contains movie ratings from users. The data preprocessing pipeline:

1. Cleans and transforms raw data
2. Extracts movie features (year, genres)
3. Creates TF-IDF vectors for content-based filtering
4. Computes similarity matrices
5. Optionally enriches data with TMDB API

## Algorithms

### Content-Based Filtering

Recommends movies based on similarity of movie features (genres, keywords, etc.). This approach analyzes movie attributes to find similar items that a user might enjoy based on their past preferences.

### Collaborative Filtering

Recommends movies based on the preferences of similar users. This approach comes in two variants:

- **User-Based**: Finds users with similar taste and recommends what they liked
- **Item-Based**: Finds relationships between items based on user rating patterns

### Matrix Factorization

Uses Singular Value Decomposition (SVD) to identify latent factors that explain the observed ratings. This technique is particularly effective for sparse datasets.

### Hybrid Approach

Combines multiple algorithms to leverage the strengths of each approach, resulting in more accurate and diverse recommendations.

## Evaluation Metrics

The system evaluates recommendations using:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Precision@K
- Recall@K
- MAP (Mean Average Precision)
- nDCG (Normalized Discounted Cumulative Gain)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [TMDB](https://www.themoviedb.org/) for the movie API
- [Streamlit](https://streamlit.io/) for the web framework 