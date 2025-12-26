"""
Quick test script to verify everything works
Run: python test_system.py
"""

import requests
import time
import subprocess
import sys

def test_files_exist():
    """Check if data files exist"""
    print("Checking data files...")
    import os
    files = ['movies.csv', 'users.csv', 'ratings.csv']
    for f in files:
        if os.path.exists(f):
            print(f"   {f} found")
        else:
            print(f"   {f} not found - run generate_data.py first!")
            return False
    return True

def test_models():
    """Test models can be loaded"""
    print("\nTesting models...")
    try:
        from models import SimpleRecommender, MatrixFactorization
        import pandas as pd
        
        ratings_df = pd.read_csv('ratings.csv')
        
        print("   Training Simple Recommender...")
        simple = SimpleRecommender(ratings_df)
        recs = simple.recommend(5, n=5)
        print(f"   Got {len(recs)} recommendations")
        
        print("   Training Matrix Factorization...")
        mf = MatrixFactorization(n_factors=10)
        mf.fit(ratings_df)
        recs = mf.recommend(5, n=5)
        print(f"   Got {len(recs)} recommendations")
        
        return True
    except Exception as e:
        print(f"   Error: {e}")
        return False

def test_api():
    """Test API endpoints"""
    print("\nTesting API...")
    print("   NOTE: Make sure API is running (python api.py)")
    
    base_url = "http://localhost:8000"
    
    try:
        # yest health
        print("   Checking /health...")
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200:
            print(f"   Health check passed")
        else:
            print(f"   Health check returned {r.status_code}")
        
        # test movies
        print("   Checking /movies...")
        r = requests.get(f"{base_url}/movies?limit=5", timeout=2)
        if r.status_code == 200:
            movies = r.json()
            print(f"   Got {len(movies)} movies")
        
        # test recs
        print("   Checking /recommend...")
        r = requests.post(
            f"{base_url}/recommend",
            json={"user_id": 5, "n": 5},
            timeout=2
        )
        if r.status_code == 200:
            data = r.json()
            print(f"   Got {len(data['recommendations'])} recommendations")
            print("\n   Sample recommendations:")
            for i, rec in enumerate(data['recommendations'][:3], 1):
                print(f"      {i}. {rec['title']} ({rec['genre']})")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("   Cannot connect to API - is it running?")
        print("      Start it with: python api.py")
        return False
    except Exception as e:
        print(f"   Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Movie Recommendation System\n")
    print("="*50)
    
    # Test data files
    if not test_files_exist():
        print("\nData files missing. Run: python generate_data.py")
        sys.exit(1)
    
    # Test models
    if not test_models():
        print("\nModels failed. Check error above.")
        sys.exit(1)
    
    # Test API
    test_api()
    
    print("\n" + "="*50)
    print("System test complete!")
    print("\nNext steps:")
    print("   1. If API isn't running: python api.py")
    print("   2. Visit http://localhost:8000/docs")
    print("   3. Try the interactive API documentation")