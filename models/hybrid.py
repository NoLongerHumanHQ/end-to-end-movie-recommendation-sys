import numpy as np
import pandas as pd

class HybridRecommender:
    """
    Hybrid recommender that combines multiple recommendation algorithms.
    
    This recommender combines the results of content-based filtering and collaborative
    filtering to provide more robust and diverse recommendations.
    """
    
    def __init__(self, content_based_recommender=None, collaborative_recommender=None, 
                matrix_factorization_recommender=None, weights=None):
        """
        Initialize the hybrid recommender.
        
        Args:
            content_based_recommender: Content-based filtering recommender
            collaborative_recommender: Collaborative filtering recommender
            matrix_factorization_recommender: Matrix factorization recommender
            weights (dict, optional): Weights for each algorithm (default: equal weights)
        """
        self.content_based_recommender = content_based_recommender
        self.collaborative_recommender = collaborative_recommender
        self.matrix_factorization_recommender = matrix_factorization_recommender
        
        # Set default weights if not provided
        if weights is None:
            self.weights = {
                'content_based': 1.0,
                'collaborative': 1.0,
                'matrix_factorization': 1.0
            }
        else:
            self.weights = weights
        
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total_weight
    
    def recommend_for_movie(self, movie_id, top_n=10):
        """
        Recommend movies similar to the given movie using a hybrid approach.
        
        Args:
            movie_id (int): ID of the movie to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with scores
        """
        recommendations = {}
        final_scores = {}
        
        # Get content-based recommendations
        if self.content_based_recommender is not None:
            cb_recs = self.content_based_recommender.recommend(movie_id, top_n=top_n*2)
            if not cb_recs.empty:
                for _, row in cb_recs.iterrows():
                    movie_id = row['movieId']
                    if movie_id not in final_scores:
                        final_scores[movie_id] = 0
                    final_scores[movie_id] += row['similarity_score'] * self.weights['content_based']
                    recommendations[movie_id] = row
        
        # Sort by final score and get top N
        top_movie_ids = [k for k, v in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]
        
        # Create final recommendations dataframe
        results = []
        for movie_id in top_movie_ids:
            rec = recommendations[movie_id].to_dict()
            rec['final_score'] = final_scores[movie_id]
            results.append(rec)
            
        return pd.DataFrame(results)
    
    def recommend_for_user(self, user_id, top_n=10):
        """
        Recommend movies for a user using a hybrid approach.
        
        Args:
            user_id (int): ID of the user to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with scores
        """
        recommendations = {}
        final_scores = {}
        
        # Get collaborative filtering recommendations
        if self.collaborative_recommender is not None:
            cf_recs = self.collaborative_recommender.recommend(user_id, top_n=top_n*2)
            if not cf_recs.empty:
                for _, row in cf_recs.iterrows():
                    movie_id = row['movieId']
                    if movie_id not in final_scores:
                        final_scores[movie_id] = 0
                    final_scores[movie_id] += row['score'] * self.weights['collaborative']
                    recommendations[movie_id] = row
        
        # Get matrix factorization recommendations
        if self.matrix_factorization_recommender is not None:
            mf_recs = self.matrix_factorization_recommender.recommend(user_id, top_n=top_n*2)
            if not mf_recs.empty:
                for _, row in mf_recs.iterrows():
                    movie_id = row['movieId']
                    if movie_id not in final_scores:
                        final_scores[movie_id] = 0
                    # Normalize predicted rating to 0-1 scale (assuming 1-5 rating scale)
                    normalized_score = (row['predicted_rating'] - 1) / 4
                    final_scores[movie_id] += normalized_score * self.weights['matrix_factorization']
                    if movie_id not in recommendations:
                        recommendations[movie_id] = row
        
        # Sort by final score and get top N
        top_movie_ids = [k for k, v in sorted(final_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]
        
        # Create final recommendations dataframe with movie details
        results = []
        for movie_id in top_movie_ids:
            rec = {'movieId': movie_id, 'final_score': final_scores[movie_id]}
            results.append(rec)
        
        return pd.DataFrame(results)
    
    def recommend(self, user_id=None, movie_id=None, user_ratings=None, top_n=10):
        """
        General recommendation method that handles different scenarios.
        
        Args:
            user_id (int, optional): ID of the user to get recommendations for
            movie_id (int, optional): ID of the movie to get recommendations for
            user_ratings (dict, optional): Dictionary mapping movie_ids to ratings
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with scores
        """
        # If we have a user ID, use user-based recommendation
        if user_id is not None:
            return self.recommend_for_user(user_id, top_n)
        
        # If we have a movie ID, use item-based recommendation
        elif movie_id is not None:
            return self.recommend_for_movie(movie_id, top_n)
            
        # If we have user ratings but no user ID, use content-based user profile
        elif user_ratings is not None and self.content_based_recommender is not None:
            movie_ids = list(user_ratings.keys())
            ratings = list(user_ratings.values())
            return self.content_based_recommender.recommend_for_user_profile(movie_ids, ratings, top_n)
            
        # If we don't have enough information, return an empty dataframe
        else:
            return pd.DataFrame() 