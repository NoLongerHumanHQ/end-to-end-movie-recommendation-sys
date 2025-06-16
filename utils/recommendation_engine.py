import os
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path

# Import recommender models
from models.content_based import ContentBasedRecommender
from models.hybrid import HybridRecommender

class RecommendationEngine:
    """
    Main recommendation engine that integrates different recommendation algorithms
    and provides a unified interface for the Streamlit application.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the recommendation engine.
        
        Args:
            data_dir (str, optional): Directory containing data files
        """
        self.data_dir = data_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.models = {}
        self.movies_df = None
        self.ratings_df = None
        self.similarity_matrix = None
        self.tfidf_matrix = None
        
        # Load data
        self.load_data()
        
        # Initialize recommenders
        self._initialize_recommenders()
        
    def load_data(self):
        """Load movie and ratings data."""
        try:
            # Try to load processed data
            processed_movies_path = os.path.join(self.processed_dir, 'processed_movies.csv')
            if os.path.exists(processed_movies_path):
                self.movies_df = pd.read_csv(processed_movies_path)
                
                # Also load processed ratings
                processed_ratings_path = os.path.join(self.processed_dir, 'processed_ratings.csv')
                if os.path.exists(processed_ratings_path):
                    self.ratings_df = pd.read_csv(processed_ratings_path)
            else:
                # Fall back to raw data
                movies_path = os.path.join(self.data_dir, 'movies.csv')
                if os.path.exists(movies_path):
                    self.movies_df = pd.read_csv(movies_path)
                
                # Load raw ratings
                ratings_path = os.path.join(self.data_dir, 'ratings.csv')
                if os.path.exists(ratings_path):
                    self.ratings_df = pd.read_csv(ratings_path)
            
            # Try to load similarity matrix if it exists
            similarity_path = os.path.join(self.processed_dir, 'similarity_matrix.npy')
            if os.path.exists(similarity_path):
                self.similarity_matrix = np.load(similarity_path)
                
        except Exception as e:
            print(f"Error loading data: {e}")
        
    def _initialize_recommenders(self):
        """Initialize different recommendation algorithms."""
        # Initialize content-based recommender
        self.models['content_based'] = ContentBasedRecommender(
            movies_df=self.movies_df,
            similarity_matrix=self.similarity_matrix,
            feature_matrix=self.tfidf_matrix
        )
        
        # Try to load pre-trained collaborative filtering models
        try:
            # Check for UserBasedCF model
            user_cf_path = os.path.join(self.processed_dir, 'user_cf.pkl')
            if os.path.exists(user_cf_path):
                self.models['user_cf'] = joblib.load(user_cf_path)
                
            # Check for ItemBasedCF model
            item_cf_path = os.path.join(self.processed_dir, 'item_cf.pkl')
            if os.path.exists(item_cf_path):
                self.models['item_cf'] = joblib.load(item_cf_path)
                
            # Check for MatrixFactorizationCF model
            mf_cf_path = os.path.join(self.processed_dir, 'matrix_factorization.pkl')
            if os.path.exists(mf_cf_path):
                self.models['matrix_factorization'] = joblib.load(mf_cf_path)
        except Exception as e:
            print(f"Error loading pre-trained models: {e}")
            
        # Initialize hybrid recommender
        self.models['hybrid'] = HybridRecommender(
            content_based_recommender=self.models.get('content_based'),
            collaborative_recommender=self.models.get('item_cf'),
            matrix_factorization_recommender=self.models.get('matrix_factorization')
        )
        
    def get_movie_by_id(self, movie_id):
        """
        Get movie details by ID.
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            pandas.Series or None: Movie details if found, None otherwise
        """
        if self.movies_df is None:
            return None
            
        movie = self.movies_df[self.movies_df['movieId'] == movie_id]
        if len(movie) == 0:
            return None
            
        return movie.iloc[0]
    
    def get_movie_by_title(self, title):
        """
        Get movie details by title (exact or partial match).
        
        Args:
            title (str): Movie title to search for
            
        Returns:
            pandas.Series or None: Movie details if found, None otherwise
        """
        if self.movies_df is None:
            return None
            
        # Try exact match first
        movie = self.movies_df[self.movies_df['title'] == title]
        if len(movie) > 0:
            return movie.iloc[0]
            
        # Try partial match
        movie = self.movies_df[self.movies_df['title'].str.contains(title, case=False)]
        if len(movie) > 0:
            return movie.iloc[0]
            
        return None
    
    def get_popular_movies(self, n=10):
        """
        Get popular movies based on ratings.
        
        Args:
            n (int): Number of movies to return
            
        Returns:
            DataFrame: Top N popular movies
        """
        if self.movies_df is None or self.ratings_df is None:
            return pd.DataFrame()
            
        # Calculate average rating and count for each movie
        ratings_stats = self.ratings_df.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            rating_mean=('rating', 'mean')
        ).reset_index()
        
        # Filter movies with at least 50 ratings
        min_ratings = 50
        popular = ratings_stats[ratings_stats['rating_count'] >= min_ratings]
        
        # Sort by average rating (descending) and select top N
        popular = popular.sort_values('rating_mean', ascending=False).head(n)
        
        # Merge with movie details
        popular_movies = pd.merge(popular, self.movies_df, on='movieId')
        
        return popular_movies
    
    def get_recent_movies(self, n=10, min_year=2010):
        """
        Get recent movies.
        
        Args:
            n (int): Number of movies to return
            min_year (int): Minimum release year
            
        Returns:
            DataFrame: Top N recent movies
        """
        if self.movies_df is None:
            return pd.DataFrame()
            
        # Filter by year if available
        if 'year' in self.movies_df.columns:
            recent_movies = self.movies_df[self.movies_df['year'] >= min_year]
        else:
            # Try to extract year from title
            recent_movies = self.movies_df[self.movies_df['title'].str.extract(r'\((\d{4})\)')[0].astype(float) >= min_year]
            
        # Sort by year (descending) and select top N
        if 'year' in recent_movies.columns:
            recent_movies = recent_movies.sort_values('year', ascending=False).head(n)
        else:
            recent_movies = recent_movies.sort_values('title', ascending=False).head(n)
            
        return recent_movies
    
    def get_recommendations(self, movie_id=None, user_id=None, n=10, algorithm='content_based'):
        """
        Get movie recommendations using the specified algorithm.
        
        Args:
            movie_id (int, optional): Movie ID for content-based recommendations
            user_id (int, optional): User ID for collaborative filtering recommendations
            n (int): Number of recommendations to return
            algorithm (str): Algorithm to use ('content_based', 'user_cf', 'item_cf', 'matrix_factorization', or 'hybrid')
            
        Returns:
            DataFrame: Recommended movies
        """
        # If no recommender is available for the requested algorithm, fall back to content-based
        if algorithm not in self.models:
            algorithm = 'content_based'
            
        recommender = self.models[algorithm]
        
        # Get recommendations
        if user_id is not None:
            recommendations = recommender.recommend(user_id=user_id, top_n=n)
        elif movie_id is not None:
            recommendations = recommender.recommend(movie_id=movie_id, top_n=n)
        else:
            # If no inputs are provided, return popular movies
            return self.get_popular_movies(n)
            
        # If no recommendations were found, return popular movies
        if recommendations.empty:
            return self.get_popular_movies(n)
            
        # Add movie details
        result = pd.merge(recommendations, self.movies_df, on='movieId', how='left')
        
        return result
    
    def search_movies(self, query, n=10):
        """
        Search for movies by title or genre.
        
        Args:
            query (str): Search query
            n (int): Maximum number of results to return
            
        Returns:
            DataFrame: Search results
        """
        if self.movies_df is None:
            return pd.DataFrame()
            
        # Search in title
        title_matches = self.movies_df[self.movies_df['title'].str.contains(query, case=False)]
        
        # Search in genres
        genre_matches = self.movies_df[self.movies_df['genres'].str.contains(query, case=False)]
        
        # Combine results (remove duplicates)
        results = pd.concat([title_matches, genre_matches]).drop_duplicates().head(n)
        
        return results
    
    def get_movie_details(self, movie_id):
        """
        Get detailed information about a movie.
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            dict or None: Movie details if found, None otherwise
        """
        movie = self.get_movie_by_id(movie_id)
        if movie is None:
            return None
            
        # Add additional data
        
        # Get average rating
        avg_rating = None
        if self.ratings_df is not None:
            movie_ratings = self.ratings_df[self.ratings_df['movieId'] == movie_id]
            if len(movie_ratings) > 0:
                avg_rating = movie_ratings['rating'].mean()
                
        if avg_rating is not None:
            movie['average_rating'] = round(avg_rating, 1)
        else:
            movie['average_rating'] = None
            
        # Get number of ratings
        movie['num_ratings'] = len(movie_ratings) if self.ratings_df is not None else 0
        
        return movie
    
    def get_similar_users(self, user_id, n=10):
        """
        Get users similar to the given user.
        
        Args:
            user_id (int): User ID
            n (int): Number of similar users to return
            
        Returns:
            list or None: List of similar user IDs if found, None otherwise
        """
        if 'user_cf' not in self.models:
            return None
            
        # Get user similarity matrix
        recommender = self.models['user_cf']
        user_similarity = recommender.compute_user_similarity()
        
        # Check if user exists
        if user_id not in recommender.user_item_matrix.index:
            return None
            
        # Get user index
        user_idx = list(recommender.user_item_matrix.index).index(user_id)
        
        # Get similarity scores
        similarity_scores = list(enumerate(user_similarity[user_idx]))
        
        # Sort by similarity score (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N user indices (excluding the input user)
        similar_user_indices = [i[0] for i in similarity_scores if i[0] != user_idx][:n]
        
        # Convert indices to user IDs
        similar_users = [recommender.user_item_matrix.index[idx] for idx in similar_user_indices]
        
        return similar_users

    def get_similar_movies(self, movie_id, n=10):
        """
        Get movies similar to the given movie using content-based similarity.
        
        Args:
            movie_id (int): Movie ID
            n (int): Number of similar movies to return
            
        Returns:
            DataFrame: Similar movies
        """
        if self.movies_df is None or self.similarity_matrix is None:
            return pd.DataFrame()
            
        # Get the index of the movie in the dataframe
        movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
        if len(movie_idx) == 0:
            return pd.DataFrame()
            
        movie_idx = movie_idx[0]
        
        # Get similarity scores for the movie
        sim_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort movies by similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar movie indices (excluding the input movie)
        sim_scores = sim_scores[1:n+1]
        movie_indices = [i[0] for i in sim_scores]
        
        # Return similar movies
        similar_movies = self.movies_df.iloc[movie_indices].copy()
        
        # Add similarity scores
        similarity_values = [i[1] for i in sim_scores]
        similar_movies['similarity'] = similarity_values
        
        return similar_movies
        
    def recommend_for_user(self, user_ratings, n=10):
        """
        Get recommendations for a user based on their ratings.
        
        Args:
            user_ratings (dict): Dictionary mapping movie IDs to ratings
            n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Recommended movies
        """
        if self.movies_df is None or self.similarity_matrix is None:
            return pd.DataFrame()
            
        if not user_ratings:
            return self.get_popular_movies(n)
            
        # Create a list of (movie_id, rating) tuples
        rated_movies = list(user_ratings.items())
        
        # Calculate weighted ratings
        all_movies = np.zeros(len(self.movies_df))
        weights = np.zeros(len(self.movies_df))
        
        for movie_id, rating in rated_movies:
            # Get the index of the movie
            movie_idx = self.movies_df[self.movies_df['movieId'] == movie_id].index
            if len(movie_idx) > 0:
                # Get similarity scores for the movie
                idx = movie_idx[0]
                sim_scores = self.similarity_matrix[idx]
                
                # Normalize rating to -1 to 1 scale (assuming 1-5 scale)
                norm_rating = (rating - 3) / 2
                
                # Add weighted scores
                all_movies += sim_scores * norm_rating
                weights += np.abs(sim_scores)
        
        # Avoid division by zero
        weights[weights == 0] = 1
        
        # Calculate weighted average
        weighted_scores = all_movies / weights
        
        # Create a list of (index, score) tuples
        movie_scores = list(enumerate(weighted_scores))
        
        # Get indices of movies that the user has already rated
        rated_indices = [self.movies_df[self.movies_df['movieId'] == movie_id].index[0]
                         for movie_id, _ in rated_movies
                         if len(self.movies_df[self.movies_df['movieId'] == movie_id].index) > 0]
        
        # Filter out already rated movies
        movie_scores = [x for x in movie_scores if x[0] not in rated_indices]
        
        # Sort by score (descending)
        movie_scores = sorted(movie_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N movies
        movie_scores = movie_scores[:n]
        movie_indices = [i[0] for i in movie_scores]
        
        # Return recommended movies
        recommendations = self.movies_df.iloc[movie_indices].copy()
        
        # Add recommendation scores
        recommendation_values = [i[1] for i in movie_scores]
        recommendations['score'] = recommendation_values 