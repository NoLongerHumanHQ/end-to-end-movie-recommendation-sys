import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
from PIL import Image
import io
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.abspath('.'))

# Import local modules
from utils.recommendation_engine import RecommendationEngine
from utils.api_handlers import get_movie_poster_url, get_movie_details, get_movie_cast
from utils.data_processing import process_movies_data

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("🎬 Movie Recommender")
page = st.sidebar.selectbox(
    "Select a page:",
    ["Home", "Movie Search", "Recommendations", "User Profile", "About"]
)

# ===== UTILITY FUNCTIONS =====

@st.cache_resource
def get_recommendation_engine():
    """Initialize and return the recommendation engine"""
    try:
        return RecommendationEngine()
    except Exception as e:
        st.error(f"Error initializing recommendation engine: {e}")
        return None

@st.cache_data
def get_poster_image(tmdb_id):
    """Get movie poster from TMDB API"""
    if not tmdb_id:
        return None
    
    poster_url = get_movie_poster_url(tmdb_id)
    if not poster_url:
        return None
        
    try:
        response = requests.get(poster_url)
        img = Image.open(io.BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Error getting poster image: {e}")
        return None

# Initialize session state for user data
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}  # {movie_id: rating}

if 'username' not in st.session_state:
    st.session_state.username = "Guest"

# Load recommendation engine
recommendation_engine = get_recommendation_engine()

# Check if engine loaded successfully
if recommendation_engine is None:
    st.warning("Error loading the recommendation system. Please check the data files.")
    st.stop()

# ===== COMPONENTS =====

def display_movie_card(movie, show_rating=False, show_recommend_button=False):
    """Display a movie card with poster and information"""
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Try to get poster from TMDB if available
        if 'tmdb_id' in movie and movie['tmdb_id']:
            poster_img = get_poster_image(movie['tmdb_id'])
            if poster_img:
                st.image(poster_img, width=200)
            else:
                st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
        else:
            st.image("https://via.placeholder.com/200x300?text=No+Poster", width=200)
    
    with col2:
        st.subheader(movie['title'])
        
        # Display year if available
        if 'year' in movie and movie['year']:
            st.write(f"**Year:** {movie['year']}")
        
        # Display genres
        st.write(f"**Genres:** {movie['genres']}")
        
        # Display rating if available
        if 'rating_mean' in movie:
            st.write(f"**Average Rating:** {movie['rating_mean']:.1f}/5.0 ({movie.get('rating_count', 0)} ratings)")
        
        # Display overview if available
        if 'overview' in movie and movie['overview']:
            st.write(movie['overview'])
        
        # Rating controls
        if show_rating:
            movie_id = movie['movieId']
            current_rating = st.session_state.user_ratings.get(movie_id, 0)
            
            new_rating = st.slider(
                "Your Rating:",
                min_value=0.0, max_value=5.0, value=float(current_rating), step=0.5,
                key=f"rating_{movie_id}"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Rating", key=f"save_{movie_id}"):
                    if new_rating > 0:
                        st.session_state.user_ratings[movie_id] = new_rating
                        st.success(f"Rating saved: {new_rating}/5.0")
                    else:
                        if movie_id in st.session_state.user_ratings:
                            del st.session_state.user_ratings[movie_id]
                        st.info("Rating removed")
                    
                    # Force a rerun to update UI
                    st.experimental_rerun()
            
            with col2:
                if current_rating > 0 and st.button("Remove Rating", key=f"remove_{movie_id}"):
                    if movie_id in st.session_state.user_ratings:
                        del st.session_state.user_ratings[movie_id]
                    st.info("Rating removed")
                    
                    # Force a rerun to update UI
                    st.experimental_rerun()
        
        # Recommend button
        if show_recommend_button:
            if st.button("Find Similar Movies", key=f"similar_{movie['movieId']}"):
                st.session_state.selected_movie = movie['movieId']
                
                # Switch to recommendations page
                st.session_state.page = "Recommendations"
                st.experimental_rerun()

# ===== PAGE CONTENT =====

# Home Page
if page == "Home":
    st.title("🎬 Welcome to the Movie Recommender System")
    st.write("""
    Discover movies you'll love with our recommendation system! This application helps you
    find movies based on your preferences using advanced recommendation algorithms.
    
    ### How to use:
    1. **Search** for movies you've watched
    2. **Rate** them based on how much you enjoyed them
    3. Get personalized **recommendations** based on your ratings
    
    You can also find movies similar to your favorites using content-based filtering.
    """)
    
    st.subheader("Popular Movies")
    popular_movies = recommendation_engine.get_popular_movies(n=6)
    
    # Display popular movies in a 2x3 grid
    for i in range(0, len(popular_movies), 3):
        cols = st.columns(3)
        for j in range(3):
            if i+j < len(popular_movies):
                with cols[j]:
                    movie = popular_movies.iloc[i+j]
                    st.subheader(movie['title'])
                    
                    # Show poster if available
                    if 'tmdb_id' in movie and movie['tmdb_id']:
                        poster_img = get_poster_image(movie['tmdb_id'])
                        if poster_img:
                            st.image(poster_img, width=200)
                    
                    st.write(f"**Rating:** {movie['rating_mean']:.1f}/5.0")
                    st.write(f"**Genres:** {movie['genres']}")
                    
                    if st.button("More Details", key=f"details_{movie['movieId']}"):
                        st.session_state.movie_details = movie['movieId']
                        st.experimental_rerun()

# Movie Search Page
elif page == "Movie Search":
    st.title("🔍 Find Movies")
    
    # Search form
    search_query = st.text_input("Search for a movie by title or genre:", placeholder="e.g. Toy Story, Comedy")
    
    if search_query:
        # Get search results
        results = recommendation_engine.search_movies(search_query, n=10)
        
        if not results.empty:
            st.write(f"Found {len(results)} movies matching '{search_query}':")
            
            # Display search results
            for idx, movie in results.iterrows():
                st.markdown("---")
                display_movie_card(movie, show_rating=True, show_recommend_button=True)
        else:
            st.info(f"No movies found matching '{search_query}'")

# Recommendations Page
elif page == "Recommendations":
    st.title("🎯 Movie Recommendations")
    
    # Choose recommendation method
    rec_method = st.selectbox(
        "Recommendation method:",
        ["Based on your ratings", "Based on a movie", "Popular movies"]
    )
    
    if rec_method == "Based on your ratings":
        # Check if user has rated any movies
        if not st.session_state.user_ratings:
            st.info("You haven't rated any movies yet. Please search for movies and rate them first.")
        else:
            st.subheader(f"Personalized Recommendations (Based on {len(st.session_state.user_ratings)} ratings)")
            
            # Get recommendations
            recommendations = recommendation_engine.recommend_for_user(st.session_state.user_ratings, n=10)
            
            if not recommendations.empty:
                for _, movie in recommendations.iterrows():
                    st.markdown("---")
                    display_movie_card(movie, show_rating=True)
            else:
                st.write("Could not generate recommendations. Try rating more movies.")
    
    elif rec_method == "Based on a movie":
        # Let user select a movie
        if 'selected_movie' in st.session_state:
            movie_id = st.session_state.selected_movie
            movie = recommendation_engine.get_movie_by_id(movie_id)
            
            if movie is not None:
                st.subheader(f"Movies similar to: {movie['title']}")
                
                # Display the selected movie
                st.markdown("### Selected Movie")
                display_movie_card(movie, show_rating=True)
                
                # Get similar movies
                similar_movies = recommendation_engine.get_similar_movies(movie_id, n=5)
                
                if not similar_movies.empty:
                    st.markdown("### Recommendations")
                    for _, rec_movie in similar_movies.iterrows():
                        st.markdown("---")
                        display_movie_card(rec_movie, show_rating=True)
                else:
                    st.write("Could not find similar movies.")
            else:
                st.error("Movie not found.")
                if 'selected_movie' in st.session_state:
                    del st.session_state.selected_movie
        else:
            st.info("Select a movie from the search page to get similar recommendations.")
    
    else:  # Popular movies
        st.subheader("Popular Movies")
        popular_movies = recommendation_engine.get_popular_movies(n=10)
        
        for _, movie in popular_movies.iterrows():
            st.markdown("---")
            display_movie_card(movie, show_rating=True)

# User Profile Page
elif page == "User Profile":
    st.title("👤 Your Profile")
    
    # User name input
    username = st.text_input("Your name:", value=st.session_state.username)
    if username != st.session_state.username:
        st.session_state.username = username
        st.success(f"Welcome, {username}!")
    
    # Display user ratings
    st.subheader("Your Rated Movies")
    
    if not st.session_state.user_ratings:
        st.info("You haven't rated any movies yet.")
    else:
        # Get movies the user has rated
        rated_movie_ids = list(st.session_state.user_ratings.keys())
        rated_movies = recommendation_engine.movies_df[
            recommendation_engine.movies_df['movieId'].isin(rated_movie_ids)
        ].copy()
        
        # Add user ratings to dataframe
        rated_movies['user_rating'] = rated_movies['movieId'].map(st.session_state.user_ratings)
        
        # Sort by rating (highest first)
        rated_movies = rated_movies.sort_values('user_rating', ascending=False)
        
        for _, movie in rated_movies.iterrows():
            st.markdown("---")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(movie['title'])
                st.write(f"**Your Rating:** {movie['user_rating']}/5.0")
                st.write(f"**Genres:** {movie['genres']}")
            
            with col2:
                if st.button("Remove Rating", key=f"remove_profile_{movie['movieId']}"):
                    if movie['movieId'] in st.session_state.user_ratings:
                        del st.session_state.user_ratings[movie['movieId']]
                        st.success("Rating removed")
                        st.experimental_rerun()
        
        # Add a button to clear all ratings
        if st.button("Clear All Ratings"):
            st.session_state.user_ratings = {}
            st.success("All ratings have been cleared!")
            st.experimental_rerun()

# About Page
elif page == "About":
    st.title("ℹ️ About This System")
    
    st.write("""
    ## Movie Recommendation System
    
    This project demonstrates an end-to-end recommendation system built with Python and Streamlit.
    The system analyzes movie features and user preferences to suggest movies you might enjoy.
    
    ### Recommendation Algorithms
    
    The system uses multiple recommendation approaches:
    
    1. **Content-Based Filtering**: Recommends movies similar to ones you already like based on 
       movie features like genres, keywords, cast, and crew.
    
    2. **Collaborative Filtering**: Suggests movies based on patterns found in user ratings.
       These algorithms find users with similar tastes and recommend movies they enjoyed.
    
    3. **Hybrid Approach**: Combines both content-based and collaborative filtering for 
       more accurate and diverse recommendations.
    
    ### Data Sources
    
    This system uses the MovieLens dataset, which contains movie ratings collected by the 
    GroupLens Research Project at the University of Minnesota.
    
    The movie posters and additional data are fetched from The Movie Database (TMDB) API.
    
    ### Technologies Used
    
    - **Python**: Core programming language
    - **Pandas & NumPy**: Data manipulation and analysis
    - **scikit-learn**: Machine learning algorithms
    - **Streamlit**: Interactive web interface
    - **TMDB API**: Movie posters and metadata
    """)
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Movies", f"{len(recommendation_engine.movies_df):,}")
        st.metric("Unique Genres", f"{recommendation_engine.movies_df['genres'].str.split('|').explode().nunique():,}")
    
    with col2:
        st.metric("Total Ratings", f"{len(recommendation_engine.ratings_df):,}")
        st.metric("Unique Users", f"{recommendation_engine.ratings_df['userId'].nunique():,}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This recommendation system was created as a demonstration project. "
    "The MovieLens dataset is used for educational purposes."
) 