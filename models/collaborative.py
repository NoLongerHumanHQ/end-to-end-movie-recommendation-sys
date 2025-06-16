import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Collaborative filtering algorithms

class UserBasedCF:
    """
    User-Based Collaborative Filtering recommender class.
    
    This recommender finds similar users based on rating patterns and recommends
    movies that similar users liked but the target user hasn't seen yet.
    """
    
    def __init__(self, ratings_df, user_item_matrix=None):
        """
        Initialize the user-based collaborative filtering recommender.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            user_item_matrix (DataFrame, optional): Pre-computed user-item matrix
        """
        self.ratings_df = ratings_df
        self.user_item_matrix = user_item_matrix
        
        if self.user_item_matrix is None:
            self.user_item_matrix = self.create_user_item_matrix()
            
    def create_user_item_matrix(self):
        """
        Create a user-item matrix from the ratings dataframe.
        
        Returns:
            DataFrame: User-item matrix
        """
        # Create the user-item matrix
        user_item_matrix = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        return user_item_matrix
    
    def compute_user_similarity(self):
        """
        Compute user-user similarity matrix using cosine similarity.
        
        Returns:
            array: User similarity matrix
        """
        # Convert to sparse matrix for efficiency
        user_matrix_sparse = csr_matrix(self.user_item_matrix.values)
        
        # Compute cosine similarity
        user_similarity = cosine_similarity(user_matrix_sparse)
        
        return user_similarity
    
    def recommend(self, user_id, top_n=10):
        """
        Recommend movies for a user using user-based collaborative filtering.
        
        Args:
            user_id (int): ID of the user to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with scores
        """
        # Check if user exists in the matrix
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        
        # Compute user similarity
        user_similarity = self.compute_user_similarity()
        
        # Get user index
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Get user's rated movies
        user_ratings = self.user_item_matrix.iloc[user_idx].to_numpy()
        user_rated = set(np.where(user_ratings > 0)[0])
        
        # Initialize empty dictionary to store scores
        scores = {}
        
        # For each movie
        for movie_idx in range(len(self.user_item_matrix.columns)):
            # Skip if user has already rated this movie
            if movie_idx in user_rated:
                continue
                
            # Get all users who rated this movie
            movie_raters = np.where(self.user_item_matrix.iloc[:, movie_idx].to_numpy() > 0)[0]
            
            if len(movie_raters) == 0:
                continue
                
            # Get similarity scores of the current user with all users who rated this movie
            sim_scores = user_similarity[user_idx, movie_raters]
            
            # Get ratings given by these users to this movie
            rate_scores = self.user_item_matrix.iloc[movie_raters, movie_idx].to_numpy()
            
            # If no similar users rated this movie, skip
            if len(sim_scores) == 0:
                continue
                
            # Calculate weighted rating
            scores[self.user_item_matrix.columns[movie_idx]] = (
                np.sum(sim_scores * rate_scores) / np.sum(np.abs(sim_scores))
            )
        
        # Get top N movie IDs
        top_movie_ids = [k for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_n]]
        
        # Create a dataframe with recommendations
        recommendations = pd.DataFrame({
            'movieId': top_movie_ids,
            'score': [scores[movie_id] for movie_id in top_movie_ids]
        })
        
        return recommendations


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering recommender class.
    
    This recommender finds similar movies based on user rating patterns and
    recommends movies that are similar to ones the user has liked.
    """
    
    def __init__(self, ratings_df, item_similarity_matrix=None):
        """
        Initialize the item-based collaborative filtering recommender.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            item_similarity_matrix (array, optional): Pre-computed item similarity matrix
        """
        self.ratings_df = ratings_df
        self.item_similarity_matrix = item_similarity_matrix
        self.user_item_matrix = self.create_user_item_matrix()
        
        # Create mapping of movie IDs to matrix indices
        self.movie_to_idx = {movie_id: idx for idx, movie_id in 
                            enumerate(self.user_item_matrix.columns)}
        self.idx_to_movie = {idx: movie_id for movie_id, idx in self.movie_to_idx.items()}
        
        if self.item_similarity_matrix is None:
            self.item_similarity_matrix = self.compute_item_similarity()
    
    def create_user_item_matrix(self):
        """
        Create a user-item matrix from the ratings dataframe.
        
        Returns:
            DataFrame: User-item matrix
        """
        # Create the user-item matrix
        user_item_matrix = self.ratings_df.pivot(
            index='userId',
            columns='movieId',
            values='rating'
        ).fillna(0)
        
        return user_item_matrix
    
    def compute_item_similarity(self):
        """
        Compute item-item similarity matrix using cosine similarity.
        
        Returns:
            array: Item similarity matrix
        """
        # Convert to sparse matrix for efficiency
        item_matrix_sparse = csr_matrix(self.user_item_matrix.values.T)
        
        # Compute cosine similarity
        item_similarity = cosine_similarity(item_matrix_sparse)
        
        return item_similarity
    
    def recommend(self, user_id, top_n=10):
        """
        Recommend movies for a user using item-based collaborative filtering.
        
        Args:
            user_id (int): ID of the user to get recommendations for
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with scores
        """
        # Check if user exists in the matrix
        if user_id not in self.user_item_matrix.index:
            return pd.DataFrame()
        
        # Get user's ratings
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_ratings = self.user_item_matrix.iloc[user_idx].to_numpy()
        
        # Find non-zero ratings (movies the user has rated)
        rated_items = np.where(user_ratings > 0)[0]
        
        # Initialize scores for all items
        scores = np.zeros(len(self.user_item_matrix.columns))
        weights = np.zeros(len(self.user_item_matrix.columns))
        
        # For each item the user has rated
        for idx in rated_items:
            # Get similarity scores
            similarity = self.item_similarity_matrix[idx]
            
            # Add weighted ratings
            scores += similarity * user_ratings[idx]
            weights += np.abs(similarity)
        
        # Normalize scores
        weights[weights == 0] = 1.0  # Avoid division by zero
        scores = scores / weights
        
        # Set scores of already rated items to 0
        scores[rated_items] = 0
        
        # Get top N item indices
        top_indices = scores.argsort()[-top_n:][::-1]
        
        # Create a dataframe with recommendations
        recommendations = pd.DataFrame({
            'movieId': [self.idx_to_movie[idx] for idx in top_indices],
            'score': [scores[idx] for idx in top_indices]
        })
        
        return recommendations


class MatrixFactorizationCF:
    """
    Matrix Factorization-based Collaborative Filtering recommender class.
    
    This recommender uses techniques like SVD to decompose the user-item matrix
    and make predictions.
    """
    
    def __init__(self, ratings_df, n_factors=100, random_state=42):
        """
        Initialize the matrix factorization recommender.
        
        Args:
            ratings_df (DataFrame): DataFrame containing user ratings
            n_factors (int): Number of latent factors
            random_state (int): Random seed for reproducibility
        """
        self.ratings_df = ratings_df
        self.n_factors = n_factors
        self.random_state = random_state
        self.model = None
        
        # Create a reader object
        self.reader = Reader(rating_scale=(0.5, 5))
        
        # Create a dataset from ratings dataframe
        self.data = Dataset.load_from_df(
            self.ratings_df[['userId', 'movieId', 'rating']], 
            self.reader
        )
        
        self._fit()
    
    def _fit(self):
        """
        Fit the SVD model to the ratings data.
        """
        # Build full trainset
        trainset = self.data.build_full_trainset()
        
        # Create and train the model
        self.model = SVD(n_factors=self.n_factors, random_state=self.random_state)
        self.model.fit(trainset)
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict the rating a user would give to a movie.
        
        Args:
            user_id (int): User ID
            movie_id (int): Movie ID
            
        Returns:
            float: Predicted rating
        """
        if self.model is None:
            self._fit()
            
        return self.model.predict(user_id, movie_id).est
    
    def recommend(self, user_id, movie_ids=None, top_n=10):
        """
        Recommend movies for a user using matrix factorization.
        
        Args:
            user_id (int): ID of the user to get recommendations for
            movie_ids (list, optional): List of movie IDs to consider
            top_n (int): Number of recommendations to return
            
        Returns:
            DataFrame: Top N recommended movies with predicted ratings
        """
        if self.model is None:
            self._fit()
            
        # If no movie_ids are provided, use all movie IDs in the dataset
        if movie_ids is None:
            movie_ids = self.ratings_df['movieId'].unique()
            
        # Get movies already rated by the user
        rated_movies = self.ratings_df[self.ratings_df['userId'] == user_id]['movieId'].values
        
        # Filter out already rated movies
        movie_ids = [m for m in movie_ids if m not in rated_movies]
        
        # Predict ratings for all movies
        predictions = [
            (movie_id, self.predict_rating(user_id, movie_id)) 
            for movie_id in movie_ids
        ]
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_recs = predictions[:top_n]
        
        # Create a dataframe with recommendations
        recommendations = pd.DataFrame(
            top_recs, 
            columns=['movieId', 'predicted_rating']
        )
        
        return recommendations 