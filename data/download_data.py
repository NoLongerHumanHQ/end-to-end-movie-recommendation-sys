import os
import requests
import zipfile
import pandas as pd
from pathlib import Path

# URL for MovieLens datasets
ML_LATEST_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
ML_25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

def download_dataset(url, output_dir=".", dataset_size="small"):
    """
    Download and extract the MovieLens dataset.
    
    Args:
        url (str): URL to download the dataset from
        output_dir (str): Directory to save the downloaded file
        dataset_size (str): Size of the dataset ("small" or "full")
        
    Returns:
        str: Path to the extracted dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get filename from URL
    filename = os.path.join(output_dir, url.split('/')[-1])
    
    # Check if zip file already exists
    if not os.path.exists(filename):
        print(f"Downloading {dataset_size} MovieLens dataset...")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        # Download with progress
        with open(filename, 'wb') as file:
            for data in response.iter_content(block_size):
                file.write(data)
                
        print(f"Download complete: {filename}")
    else:
        print(f"Using existing download: {filename}")
        
    # Extract zip file
    extract_dir = os.path.join(output_dir, f"ml-{dataset_size}")
    if os.path.exists(extract_dir):
        print(f"Dataset already extracted to: {extract_dir}")
    else:
        print(f"Extracting dataset to: {extract_dir}")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    
    # Get the actual extracted directory
    for item in os.listdir(output_dir):
        if os.path.isdir(os.path.join(output_dir, item)) and item.startswith("ml-"):
            extract_dir = os.path.join(output_dir, item)
            break
            
    return extract_dir

def copy_dataset_files(source_dir, output_dir="."):
    """
    Copy required dataset files to the output directory.
    
    Args:
        source_dir (str): Path to the extracted dataset
        output_dir (str): Directory to copy the files to
        
    Returns:
        tuple: Paths to the copied files (movies, ratings, links, tags)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Source files
    movies_src = os.path.join(source_dir, "movies.csv")
    ratings_src = os.path.join(source_dir, "ratings.csv")
    links_src = os.path.join(source_dir, "links.csv")
    tags_src = os.path.join(source_dir, "tags.csv")
    
    # Destination files
    movies_dst = os.path.join(output_dir, "movies.csv")
    ratings_dst = os.path.join(output_dir, "ratings.csv")
    links_dst = os.path.join(output_dir, "links.csv")
    tags_dst = os.path.join(output_dir, "tags.csv")
    
    # Copy files
    for src, dst in [(movies_src, movies_dst), (ratings_src, ratings_dst), 
                     (links_src, links_dst), (tags_src, tags_dst)]:
        if os.path.exists(src):
            # Read the file with pandas and write it back to ensure consistent format
            df = pd.read_csv(src)
            df.to_csv(dst, index=False)
            print(f"Copied: {dst}")
        else:
            print(f"Warning: Source file not found: {src}")
    
    return movies_dst, ratings_dst, links_dst, tags_dst

def create_sample_dataset(ratings_path, movies_path, sample_size=10000, output_dir="."):
    """
    Create a smaller sample dataset for testing.
    
    Args:
        ratings_path (str): Path to the ratings file
        movies_path (str): Path to the movies file
        sample_size (int): Number of users to include in the sample
        output_dir (str): Directory to save the sample dataset
        
    Returns:
        tuple: Paths to the sample files (movies, ratings)
    """
    print(f"Creating sample dataset with {sample_size} users...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ratings data
    ratings_df = pd.read_csv(ratings_path)
    
    # Get a sample of users
    sample_users = ratings_df['userId'].drop_duplicates().sample(n=min(sample_size, ratings_df['userId'].nunique()))
    
    # Filter ratings for sampled users
    sample_ratings = ratings_df[ratings_df['userId'].isin(sample_users)]
    
    # Get movies rated by sampled users
    sample_movies_ids = sample_ratings['movieId'].unique()
    
    # Load and filter movies data
    movies_df = pd.read_csv(movies_path)
    sample_movies = movies_df[movies_df['movieId'].isin(sample_movies_ids)]
    
    # Save sample datasets
    sample_ratings_path = os.path.join(output_dir, "sample_ratings.csv")
    sample_movies_path = os.path.join(output_dir, "sample_movies.csv")
    
    sample_ratings.to_csv(sample_ratings_path, index=False)
    sample_movies.to_csv(sample_movies_path, index=False)
    
    print(f"Sample dataset created: {sample_ratings_path}, {sample_movies_path}")
    print(f"Sample statistics: {len(sample_ratings)} ratings, {len(sample_movies)} movies, {len(sample_users)} users")
    
    return sample_movies_path, sample_ratings_path

if __name__ == "__main__":
    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ask user which dataset to download
    print("Which MovieLens dataset would you like to download?")
    print("1. Small (100,000 ratings, 9,000 movies)")
    print("2. Full (25M ratings, 62,000 movies)")
    choice = input("Enter choice (1/2) [default=1]: ").strip()
    
    if choice == "2":
        dataset_url = ML_25M_URL
        dataset_size = "25m"
    else:
        dataset_url = ML_LATEST_SMALL_URL
        dataset_size = "latest-small"
    
    # Download and extract dataset
    dataset_dir = download_dataset(dataset_url, current_dir, dataset_size)
    
    # Copy dataset files to current directory
    movies_path, ratings_path, links_path, tags_path = copy_dataset_files(dataset_dir, current_dir)
    
    # Create a smaller sample dataset if using the full dataset
    if dataset_size == "25m":
        print("\nWould you like to create a smaller sample dataset for testing?")
        create_sample = input("Create sample? (y/n) [default=y]: ").strip().lower()
        
        if create_sample != "n":
            sample_size = input("How many users to include in the sample? [default=1000]: ").strip()
            if not sample_size.isdigit():
                sample_size = 1000
            else:
                sample_size = int(sample_size)
                
            sample_movies_path, sample_ratings_path = create_sample_dataset(
                ratings_path, movies_path, sample_size, current_dir
            )
    
    print("\nDataset preparation complete!")
    print("You can now proceed with data processing and building recommenders.") 