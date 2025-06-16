import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    """
    Content-Based Filtering recommendation engine class.
    
    This recommender uses movie features (genres, tags, etc.) to find similar movies.
    """
    
    def __init__(self, movies_df, similarity_matrix=None, feature_matrix=None):
        """
        Initialize the content-based recommender.
        
        Args:
            movies_df (DataFrame): DataFrame containing movie information
            similarity_matrix (ndarray, optional): Pre-computed similarity matrix
            feature_matrix (ndarray, optional): Feature matrix used to compute similarities
        """
        self.movies_df = movies_df
        self.similarity_matrix = similarity_matrix
        self.feature_matrix = feature_matrix
        
        # Create a mapping of movie IDs to indices
        self.movie_indices = {movie_id: idx for idx, movie_id in enumerate(self.movies_df['movieId'].values)}
        self.indices_movie = {idx: movie_id for movie_id, idx in self.movie_indices.items()}
        
    def compute_similarity_matrix(self):
        """
        Compute the similarity matrix if not provided during initialization.
        """
        if self.feature_matrix is not None and self.similarity_matrix is None:
            self.similarity_matrix = cosine_similarity(self.feature_matrix)
            
    def get_movie_idx(self, movie_id):
        """Get the index for a movie ID"""
        return self.movie_indices.get(movie_id)
    
    def get_movie_id(self, idx):
        """Get the movie ID for an index"""
        return self.indices_movie.get(idx)
        
    def recommend(self, movie_id, top_n=10):
        """
        Recommend movies similar to the given movie.
        
        Args:
            movie_id (int): ID of the movie to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with similarity scores
        """
        # Ensure we have a similarity matrix
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
            
        # Get the index of the movie
        idx = self.get_movie_idx(movie_id)
        if idx is None:
            return pd.DataFrame()
            
        # Get similarity scores for all movies
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort by similarity score (descending)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n most similar movie indices (excluding the input movie)
        top_indices = [i[0] for i in similarity_scores if i[0] != idx][:top_n]
        
        # Get the movie IDs from indices
        top_movie_ids = [self.get_movie_id(idx) for idx in top_indices]
        
        # Create a dataframe with the recommended movies
        recommendations = self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)].copy()
        
        # Add similarity scores
        score_dict = {self.get_movie_id(i[0]): i[1] for i in similarity_scores if i[0] != idx}
        recommendations['similarity_score'] = recommendations['movieId'].map(score_dict)
        
        # Sort by similarity score
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations.head(top_n)
    
    def recommend_for_user_profile(self, user_rated_movies, user_ratings, top_n=10):
        """
        Recommend movies for a user based on their ratings profile.
        
        Args:
            user_rated_movies (list): List of movie IDs rated by the user
            user_ratings (list): Corresponding ratings
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies
        """
        # Ensure we have a similarity matrix
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()
            
        # Initialize an empty array for weighted average score
        weighted_avg = np.zeros(len(self.movies_df))
        
        # Calculate weighted scores
        for movie_id, rating in zip(user_rated_movies, user_ratings):
            idx = self.get_movie_idx(movie_id)
            if idx is not None:
                # Normalize rating to -1 to 1 scale (assuming 1-5 scale)
                normalized_rating = (rating - 3) / 2
                weighted_avg += self.similarity_matrix[idx] * normalized_rating
                
        # Get already watched movies
        watched_indices = [self.get_movie_idx(movie_id) for movie_id in user_rated_movies 
                           if self.get_movie_idx(movie_id) is not None]
        
        # Create a mask to exclude already watched movies
        mask = np.ones(len(self.movies_df), dtype=bool)
        mask[watched_indices] = False
        
        # Apply the mask and get top scores
        masked_scores = list(enumerate(weighted_avg * mask))
        sorted_scores = sorted(masked_scores, key=lambda x: x[1], reverse=True)
        
        # Get top n movie indices
        top_indices = [i[0] for i in sorted_scores][:top_n]
        top_movie_ids = [self.get_movie_id(idx) for idx in top_indices]
        
        # Create a dataframe with the recommended movies
        recommendations = self.movies_df[self.movies_df['movieId'].isin(top_movie_ids)].copy()
        
        # Add weighted scores
        score_dict = {self.get_movie_id(i[0]): i[1] for i in sorted_scores if i[0] in top_indices}
        recommendations['recommendation_score'] = recommendations['movieId'].map(score_dict)
        
        # Sort by recommendation score
        recommendations = recommendations.sort_values('recommendation_score', ascending=False)
        
        return recommendations.head(top_n) 