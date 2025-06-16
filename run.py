#!/usr/bin/env python
"""
Movie Recommendation System Launcher

This script provides a simple way to run the movie recommendation system.
It checks for required files and dependencies before launching the Streamlit app.
"""

import os
import sys
import subprocess
import argparse
from dotenv import load_dotenv

def check_environment():
    """Check if environment is properly set up."""
    # Check for .env file and TMDB API key
    load_dotenv()
    tmdb_api_key = os.getenv('TMDB_API_KEY')
    if not tmdb_api_key or tmdb_api_key == 'your_tmdb_api_key_here':
        print("Warning: TMDB API key not found or using default value.")
        print("Some features like movie posters may not work correctly.")
        print("Create a .env file with your TMDB API key or copy from .env.example")
        print()

def check_data():
    """Check if required data files exist."""
    # Check for processed data directory
    processed_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    raw_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Check for raw data files
    if not os.path.exists(os.path.join(raw_dir, 'movies.csv')) or \
       not os.path.exists(os.path.join(raw_dir, 'ratings.csv')):
        print("Raw MovieLens data files not found!")
        print("Please run the data download script:")
        print("  python movie_recommendation_system/data/download_data.py")
        return False
    
    # Check for processed data
    if not os.path.exists(processed_dir) or \
       not os.path.exists(os.path.join(processed_dir, 'processed_movies.csv')):
        print("Processed data not found!")
        print("Processing the MovieLens dataset...")
        
        # Run the data processing script
        try:
            subprocess.run([
                sys.executable, 
                os.path.join(os.path.dirname(__file__), 'process_data.py')
            ], check=True)
            print("Data processing completed successfully!")
        except subprocess.CalledProcessError:
            print("Error processing data. Please check the error message above.")
            return False
    
    return True

def main(args):
    """Main function to run the movie recommendation system."""
    print("Starting Movie Recommendation System...")
    
    # Check environment and data
    check_environment()
    if not check_data() and not args.force:
        print("Error: Missing required data files. Please follow the setup instructions.")
        return 1
    
    # Launch the Streamlit app
    app_path = os.path.join(os.path.dirname(__file__), 'app.py')
    print(f"Launching application from: {app_path}")
    
    try:
        subprocess.run([
            'streamlit', 'run', 
            app_path,
            '--server.headless', 'true',
            '--browser.serverAddress', '0.0.0.0'
        ])
    except FileNotFoundError:
        print("Error: Streamlit not found. Is it installed correctly?")
        print("Try running: pip install streamlit")
        return 1
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Movie Recommendation System')
    parser.add_argument('--force', action='store_true', 
                        help='Force launch even if data files are missing')
    
    args = parser.parse_args()
    sys.exit(main(args)) 