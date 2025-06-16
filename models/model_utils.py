import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from surprise import accuracy

def calculate_rmse(predictions):
    """
    Calculate Root Mean Square Error from Surprise predictions.
    
    Args:
        predictions: List of predictions from Surprise
        
    Returns:
        float: RMSE value
    """
    return accuracy.rmse(predictions, verbose=False)

def calculate_mae(predictions):
    """
    Calculate Mean Absolute Error from Surprise predictions.
    
    Args:
        predictions: List of predictions from Surprise
        
    Returns:
        float: MAE value
    """
    return accuracy.mae(predictions, verbose=False)

def precision_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate precision@k for recommendation evaluation.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Precision@k value
    """
    # Truncate the list to k items
    recommended_items = recommended_items[:k]
    
    # Count relevant items in recommendations
    hits = len(set(recommended_items) & set(relevant_items))
    
    # Calculate precision
    return hits / min(k, len(recommended_items)) if len(recommended_items) > 0 else 0

def recall_at_k(recommended_items, relevant_items, k=10):
    """
    Calculate recall@k for recommendation evaluation.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: Recall@k value
    """
    # Truncate the list to k items
    recommended_items = recommended_items[:k]
    
    # Count relevant items in recommendations
    hits = len(set(recommended_items) & set(relevant_items))
    
    # Calculate recall
    return hits / len(relevant_items) if len(relevant_items) > 0 else 0

def mean_average_precision(recommended_items, relevant_items, k=10):
    """
    Calculate Mean Average Precision (MAP) for recommendation evaluation.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        k (int): Number of recommendations to consider
        
    Returns:
        float: MAP value
    """
    # Truncate the list to k items
    recommended_items = recommended_items[:k]
    
    # Calculate average precision
    hits = 0
    sum_precisions = 0
    
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            hits += 1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
    
    return sum_precisions / len(relevant_items) if len(relevant_items) > 0 else 0

def ndcg_at_k(recommended_items, relevant_items, ratings=None, k=10):
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG) for recommendation evaluation.
    
    Args:
        recommended_items (list): List of recommended item IDs
        relevant_items (list): List of actually relevant item IDs
        ratings (dict, optional): Dictionary mapping item IDs to ratings
        k (int): Number of recommendations to consider
        
    Returns:
        float: nDCG@k value
    """
    # Truncate the list to k items
    recommended_items = recommended_items[:k]
    
    # If no ratings are provided, use binary relevance (1 if item is relevant, 0 otherwise)
    if ratings is None:
        ratings = {item: 1 for item in relevant_items}
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            relevance = ratings.get(item, 1)
            dcg += relevance / np.log2(i + 2)  # log base 2 of position + 1 (1-indexed)
    
    # Calculate ideal DCG (IDCG)
    # Sort relevant items by rating in descending order
    ideal_items = sorted(relevant_items, key=lambda x: ratings.get(x, 1), reverse=True)[:k]
    idcg = 0
    for i, item in enumerate(ideal_items):
        relevance = ratings.get(item, 1)
        idcg += relevance / np.log2(i + 2)  # log base 2 of position + 1 (1-indexed)
    
    # Calculate nDCG
    return dcg / idcg if idcg > 0 else 0

def evaluate_recommender(recommender, test_users, test_data, top_n=10):
    """
    Evaluate a recommender system using various metrics.
    
    Args:
        recommender: Recommender system to evaluate
        test_users (list): List of user IDs for testing
        test_data (DataFrame): DataFrame with test data (userId, movieId, rating)
        top_n (int): Number of recommendations to consider
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    precisions = []
    recalls = []
    maps = []
    ndcgs = []
    
    start_time = time.time()
    
    for user_id in test_users:
        # Get actual items rated positively by the user
        relevant_items = test_data[
            (test_data['userId'] == user_id) & 
            (test_data['rating'] >= 4)  # Consider ratings >= 4 as relevant
        ]['movieId'].tolist()
        
        if not relevant_items:
            continue
            
        # Get recommendations for the user
        recommendations = recommender.recommend(user_id=user_id, top_n=top_n)
        if recommendations.empty:
            continue
            
        recommended_items = recommendations['movieId'].tolist()
        
        # Calculate metrics
        precisions.append(precision_at_k(recommended_items, relevant_items, k=top_n))
        recalls.append(recall_at_k(recommended_items, relevant_items, k=top_n))
        maps.append(mean_average_precision(recommended_items, relevant_items, k=top_n))
        
        # Create ratings dictionary for nDCG calculation
        user_ratings = test_data[test_data['userId'] == user_id].set_index('movieId')['rating'].to_dict()
        ndcgs.append(ndcg_at_k(recommended_items, relevant_items, user_ratings, k=top_n))
    
    # Calculate average metrics
    avg_precision = np.mean(precisions) if precisions else 0
    avg_recall = np.mean(recalls) if recalls else 0
    avg_map = np.mean(maps) if maps else 0
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    return {
        'precision@k': avg_precision,
        'recall@k': avg_recall,
        'MAP': avg_map,
        'nDCG@k': avg_ndcg,
        'execution_time': execution_time
    }

def plot_evaluation_metrics(metrics_dict):
    """
    Plot evaluation metrics for comparison.
    
    Args:
        metrics_dict (dict): Dictionary with algorithm names as keys and metric dictionaries as values
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    algorithms = list(metrics_dict.keys())
    precision_values = [metrics_dict[alg]['precision@k'] for alg in algorithms]
    recall_values = [metrics_dict[alg]['recall@k'] for alg in algorithms]
    map_values = [metrics_dict[alg]['MAP'] for alg in algorithms]
    ndcg_values = [metrics_dict[alg]['nDCG@k'] for alg in algorithms]
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Precision plot
    axs[0, 0].bar(algorithms, precision_values)
    axs[0, 0].set_title('Precision@K')
    axs[0, 0].set_ylim(0, 1)
    
    # Recall plot
    axs[0, 1].bar(algorithms, recall_values)
    axs[0, 1].set_title('Recall@K')
    axs[0, 1].set_ylim(0, 1)
    
    # MAP plot
    axs[1, 0].bar(algorithms, map_values)
    axs[1, 0].set_title('Mean Average Precision')
    axs[1, 0].set_ylim(0, 1)
    
    # nDCG plot
    axs[1, 1].bar(algorithms, ndcg_values)
    axs[1, 1].set_title('nDCG@K')
    axs[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    return fig

def create_recommendation_df(recommendations, movies_df):
    """
    Create a DataFrame with movie details from recommendation results.
    
    Args:
        recommendations (DataFrame): DataFrame with movie IDs and scores
        movies_df (DataFrame): DataFrame with movie details
        
    Returns:
        DataFrame: Merged DataFrame with recommendations and movie details
    """
    # Merge with movie details
    result = pd.merge(recommendations, movies_df, on='movieId')
    
    # Sort by score if it exists
    if 'score' in result.columns:
        result = result.sort_values('score', ascending=False)
    elif 'similarity_score' in result.columns:
        result = result.sort_values('similarity_score', ascending=False)
    elif 'predicted_rating' in result.columns:
        result = result.sort_values('predicted_rating', ascending=False)
    elif 'final_score' in result.columns:
        result = result.sort_values('final_score', ascending=False)
    
    return result 