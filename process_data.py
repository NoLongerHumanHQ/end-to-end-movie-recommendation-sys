import os
import pandas as pd
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import local modules
from utils.data_processing import load_datasets, process_movies_data, clean_title
from utils.api_handlers import enrich_movie_data

def main(args):
    print("Starting data processing...")
    start_time = time.time()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MovieLens data
    print("Loading MovieLens dataset...")
    movies_df, ratings_df, links_df, tags_df = load_datasets(args.data_dir)
    
    # Process movies data
    print("Processing movie data...")
    movies_df = process_movies_data(movies_df)
    
    # Create TF-IDF vectors for content-based filtering
    print("Creating TF-IDF vectors for movies...")
    
    # Create combined metadata field for movies using genres
    movies_df['metadata'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))
    
    # Add titles to metadata
    movies_df['metadata'] = movies_df['clean_title'] + ' ' + movies_df['metadata']
    
    # Process tags if available
    if tags_df is not None:
        print("Processing tags data...")
        # Aggregate tags by movie
        movie_tags = tags_df.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
        # Merge with movies
        movies_df = pd.merge(movies_df, movie_tags, on='movieId', how='left')
        movies_df['tag'] = movies_df['tag'].fillna('')
        # Add tags to metadata
        movies_df['metadata'] = movies_df['metadata'] + ' ' + movies_df['tag']
    
    # Create TF-IDF matrix
    print("Creating TF-IDF matrix...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['metadata'].fillna(''))
    
    # Compute cosine similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Enrich data with TMDB if requested
    if args.enrich:
        print("Enriching data with TMDB API...")
        # Only enrich the first 1000 movies if in sample mode
        if args.sample:
            enrich_movies = movies_df.head(1000)
        else:
            enrich_movies = movies_df
        
        enriched_data = enrich_movie_data(enrich_movies)
        if isinstance(enriched_data, pd.DataFrame):
            movies_df = enriched_data
    
    # Create sample data if requested
    if args.sample:
        print("Creating sample dataset...")
        # Sample users and their ratings
        user_sample = np.random.choice(ratings_df['userId'].unique(), 
                                     size=min(100, len(ratings_df['userId'].unique())), 
                                     replace=False)
        ratings_sample = ratings_df[ratings_df['userId'].isin(user_sample)]
        
        # Get movies from the sampled ratings
        movie_ids_sample = ratings_sample['movieId'].unique()
        movies_sample = movies_df[movies_df['movieId'].isin(movie_ids_sample)]
        
        # Sample a few more popular movies to ensure we have enough
        if len(movies_sample) < 200:
            popular_movies = ratings_df['movieId'].value_counts().head(200 - len(movies_sample)).index
            more_movies = movies_df[movies_df['movieId'].isin(popular_movies)]
            movies_sample = pd.concat([movies_sample, more_movies]).drop_duplicates()
        
        # Update dataframes
        movies_df = movies_sample
        ratings_df = ratings_sample
        similarity_matrix = cosine_similarity(
            tfidf.transform(movies_df['metadata'].fillna(''))
        )
    
    # Save processed data
    print(f"Saving processed data to {args.output_dir}...")
    movies_df.to_csv(os.path.join(args.output_dir, 'processed_movies.csv'), index=False)
    ratings_df.to_csv(os.path.join(args.output_dir, 'processed_ratings.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'similarity_matrix.npy'), similarity_matrix)
    np.save(os.path.join(args.output_dir, 'tfidf_matrix.npy'), tfidf_matrix.toarray())
    
    # Save links if available
    if links_df is not None:
        links_df.to_csv(os.path.join(args.output_dir, 'processed_links.csv'), index=False)
    
    # Save tags if available
    if tags_df is not None:
        tags_df.to_csv(os.path.join(args.output_dir, 'processed_tags.csv'), index=False)
    
    print(f"Data processing completed in {time.time() - start_time:.2f} seconds!")
    print(f"Processed data saved to {args.output_dir}")
    print(f"Movies: {len(movies_df)}, Ratings: {len(ratings_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process MovieLens data for recommendation engine')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Directory containing MovieLens data files')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                        help='Directory to save processed data files')
    parser.add_argument('--sample', action='store_true',
                        help='Create a smaller sample dataset for testing')
    parser.add_argument('--enrich', action='store_true',
                        help='Enrich movie data with TMDB API')
    
    args = parser.parse_args()
    main(args) 