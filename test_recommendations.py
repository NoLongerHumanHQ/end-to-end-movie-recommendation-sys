#!/usr/bin/env python
"""
Test Recommendation Engine

This script tests the recommendation engine functionality by:
1. Loading the recommendation engine
2. Getting popular movies
3. Getting recommendations for a sample user
4. Getting similar movies to a given movie

Usage:
  python test_recommendations.py

Dependencies:
  - pandas
  - utils.recommendation_engine
"""

import os
import sys
import pandas as pd
import time

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our recommendation engine
from utils.recommendation_engine import RecommendationEngine

def test_recommendation_engine():
    """Test the main recommendation engine functionality."""
    print("Testing Recommendation Engine...")
    
    # Initialize recommendation engine
    start_time = time.time()
    print("Initializing recommendation engine...")
    engine = RecommendationEngine()
    print(f"Initialization took {time.time() - start_time:.2f} seconds")
    
    # Test 1: Get popular movies
    print("\n=== Test 1: Get Popular Movies ===")
    popular_movies = engine.get_popular_movies(n=5)
    if len(popular_movies) > 0:
        print(f"Found {len(popular_movies)} popular movies:")
        for _, movie in popular_movies.iterrows():
            print(f"  - {movie['title']} (Rating: {movie['rating_mean']:.1f}/5, {movie['rating_count']} ratings)")
        print("Test 1: PASSED")
    else:
        print("No popular movies found!")
        print("Test 1: FAILED")
    
    # Test 2: Search for a movie
    print("\n=== Test 2: Search for Movies ===")
    search_term = "toy story"
    search_results = engine.search_movies(search_term, n=3)
    if len(search_results) > 0:
        print(f"Search for '{search_term}' found {len(search_results)} results:")
        for _, movie in search_results.iterrows():
            print(f"  - {movie['title']} ({movie['genres']})")
        
        # Get the first movie ID for the next tests
        first_movie_id = search_results.iloc[0]['movieId']
        print("Test 2: PASSED")
    else:
        print(f"No movies found matching '{search_term}'!")
        print("Test 2: FAILED")
        first_movie_id = 1  # Default to movie ID 1 (usually a popular movie)
    
    # Test 3: Get similar movies
    print("\n=== Test 3: Get Similar Movies ===")
    movie = engine.get_movie_by_id(first_movie_id)
    print(f"Finding similar movies to: {movie['title']}")
    
    similar_movies = engine.get_similar_movies(first_movie_id, n=5)
    if len(similar_movies) > 0:
        print(f"Found {len(similar_movies)} similar movies:")
        for _, movie in similar_movies.iterrows():
            print(f"  - {movie['title']} (Similarity: {movie.get('similarity', 0):.3f})")
        print("Test 3: PASSED")
    else:
        print("No similar movies found!")
        print("Test 3: FAILED")
    
    # Test 4: Get user recommendations
    print("\n=== Test 4: Get User Recommendations ===")
    
    # Create sample user ratings (movie_id: rating)
    user_ratings = {
        1: 5.0,  # Usually "Toy Story"
        318: 4.5,  # Usually "Shawshank Redemption"
        296: 1.0,  # Usually "Pulp Fiction"
    }
    
    print("Sample user rated the following movies:")
    for movie_id, rating in user_ratings.items():
        movie = engine.get_movie_by_id(movie_id)
        if movie is not None:
            print(f"  - {movie['title']}: {rating}/5.0")
    
    print("\nGenerating recommendations...")
    recommendations = engine.recommend_for_user(user_ratings, n=5)
    
    if len(recommendations) > 0:
        print(f"Found {len(recommendations)} recommended movies:")
        for _, movie in recommendations.iterrows():
            print(f"  - {movie['title']} ({movie['genres']})")
        print("Test 4: PASSED")
    else:
        print("No recommendations found!")
        print("Test 4: FAILED")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_recommendation_engine() 