import requests
import os
from dotenv import load_dotenv
import time
from PIL import Image
from io import BytesIO
import pandas as pd

# Load environment variables
load_dotenv()

# TMDB API configuration
TMDB_API_KEY = os.getenv('TMDB_API_KEY')
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def get_tmdb_movie_id(movie_title, year=None):
    """
    Search for a movie in TMDB by title and year.
    
    Args:
        movie_title (str): Movie title
        year (int, optional): Release year
        
    Returns:
        int or None: TMDB movie ID if found, None otherwise
    """
    if not TMDB_API_KEY:
        return None
        
    search_url = f"{TMDB_BASE_URL}/search/movie"
    params = {
        'api_key': TMDB_API_KEY,
        'query': movie_title,
        'language': 'en-US',
        'page': 1,
        'include_adult': 'false'
    }
    
    if year:
        params['year'] = year
        
    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['results'] and len(data['results']) > 0:
            return data['results'][0]['id']
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error searching for movie: {e}")
        return None

def get_movie_details(tmdb_id):
    """
    Get detailed information for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        dict or None: Movie details if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_url = f"{TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'language': 'en-US',
        'append_to_response': 'credits,videos,keywords'
    }
    
    try:
        response = requests.get(movie_url, params=params)
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting movie details: {e}")
        return None

def get_movie_poster_url(tmdb_id):
    """
    Get the poster URL for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        str or None: Poster URL if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'poster_path' in movie_details and movie_details['poster_path']:
        return f"{TMDB_IMAGE_BASE_URL}{movie_details['poster_path']}"
    else:
        return None

def get_movie_backdrop_url(tmdb_id):
    """
    Get the backdrop URL for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        str or None: Backdrop URL if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'backdrop_path' in movie_details and movie_details['backdrop_path']:
        return f"{TMDB_IMAGE_BASE_URL}{movie_details['backdrop_path']}"
    else:
        return None

def get_movie_trailer_url(tmdb_id):
    """
    Get the YouTube trailer URL for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        str or None: YouTube URL if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'videos' in movie_details and movie_details['videos']['results']:
        videos = movie_details['videos']['results']
        trailers = [v for v in videos if v['type'] == 'Trailer' and v['site'] == 'YouTube']
        
        if trailers:
            return f"https://www.youtube.com/watch?v={trailers[0]['key']}"
            
    return None

def get_movie_cast(tmdb_id, limit=5):
    """
    Get the top cast members for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        limit (int): Maximum number of cast members to return
        
    Returns:
        list or None: List of cast members if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'credits' in movie_details and 'cast' in movie_details['credits']:
        cast = movie_details['credits']['cast']
        return cast[:limit]
        
    return None

def get_movie_director(tmdb_id):
    """
    Get the director(s) of a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        list or None: List of directors if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'credits' in movie_details and 'crew' in movie_details['credits']:
        crew = movie_details['credits']['crew']
        directors = [c for c in crew if c['job'] == 'Director']
        return directors
        
    return None

def get_movie_poster_image(tmdb_id):
    """
    Get the poster image for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        PIL.Image or None: Image if found, None otherwise
    """
    poster_url = get_movie_poster_url(tmdb_id)
    
    if not poster_url:
        return None
        
    try:
        response = requests.get(poster_url)
        response.raise_for_status()
        
        image = Image.open(BytesIO(response.content))
        return image
        
    except requests.exceptions.RequestException as e:
        print(f"Error getting poster image: {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def get_movie_keywords(tmdb_id):
    """
    Get keywords for a movie from TMDB.
    
    Args:
        tmdb_id (int): TMDB movie ID
        
    Returns:
        list or None: List of keywords if found, None otherwise
    """
    if not TMDB_API_KEY or not tmdb_id:
        return None
        
    movie_details = get_movie_details(tmdb_id)
    
    if movie_details and 'keywords' in movie_details and 'keywords' in movie_details['keywords']:
        return [k['name'] for k in movie_details['keywords']['keywords']]
        
    return None

def enrich_movie_data(movies_df):
    """
    Enrich movie data with additional information from TMDB.
    
    Args:
        movies_df (DataFrame): DataFrame with movie information
        
    Returns:
        DataFrame: Enriched DataFrame
    """
    if not TMDB_API_KEY:
        print("TMDB API key not found. Skipping movie data enrichment.")
        return movies_df
    
    # Create copies of the results DataFrame to avoid modifying the original
    result_df = movies_df.copy()
    
    # Initialize new columns
    result_df['tmdb_id'] = None
    result_df['poster_url'] = None
    result_df['backdrop_url'] = None
    result_df['overview'] = None
    result_df['tagline'] = None
    result_df['runtime'] = None
    result_df['keywords'] = None
    
    # Process each movie
    for idx, row in result_df.iterrows():
        movie_title = row.get('clean_title', row.get('title', ''))
        year = row.get('year')
        
        # Get TMDB ID
        tmdb_id = get_tmdb_movie_id(movie_title, year)
        result_df.at[idx, 'tmdb_id'] = tmdb_id
        
        if tmdb_id:
            # Get movie details
            movie_details = get_movie_details(tmdb_id)
            
            if movie_details:
                # Add poster and backdrop URLs
                if 'poster_path' in movie_details and movie_details['poster_path']:
                    result_df.at[idx, 'poster_url'] = f"{TMDB_IMAGE_BASE_URL}{movie_details['poster_path']}"
                    
                if 'backdrop_path' in movie_details and movie_details['backdrop_path']:
                    result_df.at[idx, 'backdrop_url'] = f"{TMDB_IMAGE_BASE_URL}{movie_details['backdrop_path']}"
                    
                # Add other details
                result_df.at[idx, 'overview'] = movie_details.get('overview')
                result_df.at[idx, 'tagline'] = movie_details.get('tagline')
                result_df.at[idx, 'runtime'] = movie_details.get('runtime')
                
                # Add keywords
                if 'keywords' in movie_details and 'keywords' in movie_details['keywords']:
                    result_df.at[idx, 'keywords'] = [k['name'] for k in movie_details['keywords']['keywords']]
        
        # Throttle requests to avoid hitting API rate limits
        time.sleep(0.25)
    
    return result_df 