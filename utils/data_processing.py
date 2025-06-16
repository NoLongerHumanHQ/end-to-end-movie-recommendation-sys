import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.stem.snowball import SnowballStemmer
import nltk

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_datasets(data_path):
    """
    Load MovieLens datasets from the specified path.
    
    Args:
        data_path (str): Path to the directory containing MovieLens data files
        
    Returns:
        tuple: (movies_df, ratings_df, links_df, tags_df)
    """
    movies_df = pd.read_csv(os.path.join(data_path, 'movies.csv'))
    ratings_df = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    
    # Try to load other files if they exist
    try:
        links_df = pd.read_csv(os.path.join(data_path, 'links.csv'))
    except:
        links_df = None
    
    try:
        tags_df = pd.read_csv(os.path.join(data_path, 'tags.csv'))
    except:
        tags_df = None
    
    return movies_df, ratings_df, links_df, tags_df

def clean_title(title):
    """Extract the year from the title and create a clean title."""
    # Extract year using regex pattern
    match = re.search(r'(.+)\s*\((\d{4})\)', title)
    if match:
        clean_title = match.group(1).strip()
        year = int(match.group(2))
        return clean_title, year
    return title, None

def process_movies_data(movies_df):
    """
    Process the movies dataframe by extracting years, cleaning titles, and expanding genres.
    
    Args:
        movies_df (DataFrame): The movies dataframe
        
    Returns:
        DataFrame: Processed movies dataframe
    """
    # Create copies of the clean title and year
    movies_df[['clean_title', 'year']] = pd.DataFrame(
        movies_df['title'].apply(clean_title).tolist(), 
        index=movies_df.index
    )
    
    # Convert genres from string to list
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    
    # Create dummy variables for genres
    genres_dummies = movies_df['genres_list'].explode().str.get_dummies().groupby(level=0).sum()
    
    # Add genre features to the dataframe
    movies_df = pd.concat([movies_df, genres_dummies], axis=1)
    
    return movies_df

def create_user_item_matrix(ratings_df):
    """
    Creates a user-item matrix based on ratings.
    
    Args:
        ratings_df (DataFrame): DataFrame containing user ratings
        
    Returns:
        DataFrame: User-item matrix with users as rows and movies as columns
    """
    user_item_matrix = ratings_df.pivot(
        index='userId', 
        columns='movieId', 
        values='rating'
    )
    
    return user_item_matrix

def calculate_movie_popularity(ratings_df):
    """
    Calculate popularity metrics for movies.
    
    Args:
        ratings_df (DataFrame): DataFrame containing user ratings
        
    Returns:
        DataFrame: DataFrame with popularity metrics
    """
    # Count number of ratings and calculate average rating for each movie
    movie_stats = ratings_df.groupby('movieId').agg(
        rating_count=('rating', 'count'),
        rating_mean=('rating', 'mean')
    )
    
    # Calculate minimum number of ratings required for consideration
    min_ratings = ratings_df['movieId'].value_counts().quantile(0.9)
    
    # Calculate weighted rating (IMDb formula)
    C = ratings_df['rating'].mean()
    m = min_ratings
    
    movie_stats['weighted_rating'] = (movie_stats['rating_count'] / (movie_stats['rating_count'] + m) * 
                                    movie_stats['rating_mean'] + 
                                    m / (movie_stats['rating_count'] + m) * C)
    
    return movie_stats

def create_movie_features(movies_df, tags_df=None):
    """
    Create feature vectors for movies based on available metadata.
    
    Args:
        movies_df (DataFrame): Processed movies dataframe
        tags_df (DataFrame, optional): DataFrame containing tags
        
    Returns:
        tuple: (DataFrame with features, feature matrix)
    """
    # Combine metadata for content-based filtering
    movies_df['metadata'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))
    
    if tags_df is not None:
        # Process tags if available
        tags_by_movie = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        # Merge tags with movies
        movies_df = pd.merge(movies_df, tags_by_movie, on='movieId', how='left')
        movies_df['tag'] = movies_df['tag'].fillna('')
        # Update metadata with tags
        movies_df['metadata'] = movies_df['metadata'] + ' ' + movies_df['tag']
    
    # Clean and preprocess the metadata text
    stemmer = SnowballStemmer('english')
    movies_df['metadata'] = movies_df['metadata'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.lower().split()]))
    
    # Create feature vectors using TF-IDF
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000
    )
    tfidf_matrix = tfidf.fit_transform(movies_df['metadata'])
    
    return movies_df, tfidf_matrix

def compute_similarity_matrix(matrix):
    """
    Compute the cosine similarity matrix.
    
    Args:
        matrix: Feature matrix to compute similarities for
        
    Returns:
        array: Cosine similarity matrix
    """
    return cosine_similarity(matrix)

def save_processed_data(movies_df, ratings_df, tfidf_matrix, similarity_matrix, output_path):
    """
    Save processed data to disk.
    
    Args:
        movies_df (DataFrame): Processed movies dataframe
        ratings_df (DataFrame): Processed ratings dataframe
        tfidf_matrix: TF-IDF feature matrix
        similarity_matrix: Cosine similarity matrix
        output_path (str): Path to save the processed data
        
    Returns:
        None
    """
    # Save DataFrames to CSV
    movies_df.to_csv(os.path.join(output_path, 'processed_movies.csv'), index=False)
    ratings_df.to_csv(os.path.join(output_path, 'processed_ratings.csv'), index=False)
    
    # Save matrices as numpy arrays
    np.save(os.path.join(output_path, 'tfidf_matrix.npy'), tfidf_matrix.toarray())
    np.save(os.path.join(output_path, 'similarity_matrix.npy'), similarity_matrix) 