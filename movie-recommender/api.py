"""
FastAPI Recommendation Service
Run: python api.py
or: uvicorn api:app --reload
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import uvicorn
from models import SimpleRecommender, MatrixFactorization

app = FastAPI(title="Movie Recommendation API")

# global variables
model = None
movies_df = None
ratings_df = None

class RecommendRequest(BaseModel):
    user_id: int
    n: Optional[int] = 10

@app.on_event("startup")
async def load_data():
    """Load data and train model on startup"""
    global model, movies_df, ratings_df
    
    print("Loading data...")
    try:
        movies_df = pd.read_csv('movies.csv')
        ratings_df = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        print("Data files not found! Run generate_data.py first")
        return
    
    print("Training model...")
    model = SimpleRecommender(ratings_df)
    print("Model ready!")

@app.get("/")
async def root():
    return {
        "message": "🎬 Movie Recommendation API",
        "endpoints": {
            "docs": "/docs",
            "recommend": "POST /recommend",
            "movies": "GET /movies",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "not ready",
        "model_loaded": model is not None
    }

@app.post("/recommend")
async def get_recommendations(request: RecommendRequest):
    """Get movie recommendations for a user"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        recommended_ids = model.recommend(request.user_id, n=request.n)
        
        recommendations = []
        for movie_id in recommended_ids:
            movie = movies_df[movies_df['movie_id'] == movie_id]
            if not movie.empty:
                movie = movie.iloc[0]
                recommendations.append({
                    'movie_id': int(movie['movie_id']),
                    'title': movie['title'],
                    'genre': movie['genre'],
                    'year': int(movie['year']),
                    'avg_rating': float(movie['avg_rating'])
                })
        
        return {
            'user_id': request.user_id,
            'recommendations': recommendations
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/movies")
async def get_movies(genre: Optional[str] = None, limit: int = 20):
    """Get movies, optionally filtered by genre"""
    if movies_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    df = movies_df
    if genre:
        df = df[df['genre'].str.lower() == genre.lower()]
    
    df = df.head(limit)
    return df.to_dict('records')

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if movies_df is None or ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    
    return {
        "total_movies": len(movies_df),
        "total_ratings": len(ratings_df),
        "genres": movies_df['genre'].unique().tolist()
    }

if __name__ == "__main__":
    print("Starting API server...")
    print("API docs will be at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)