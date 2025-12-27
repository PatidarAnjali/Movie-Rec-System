from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import uvicorn
from models import SimpleRecommender

app = FastAPI(title="Movie Recommendation API")
model = None
movies_df = None
ratings_df = None

class RecommendRequest(BaseModel):
    user_id: int
    n: Optional[int] = 10

@app.on_event("startup")
async def load_data():
    global model, movies_df, ratings_df
    print("Loading data...")
    movies_df = pd.read_csv('movies.csv')
    ratings_df = pd.read_csv('ratings.csv')
    print("Training model...")
    model = SimpleRecommender(ratings_df)
    print("Model ready!")

@app.get("/")
async def root():
    return {"message": "Movie Recommendation API", "docs": "/docs"}

@app.post("/recommend")
async def get_recommendations(request: RecommendRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
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

            })
    return {'user_id': request.user_id, 'recommendations': recommendations}

@app.get("/movies")
async def get_movies(limit: int = 20):
    return movies_df.head(limit).to_dict('records')

if __name__ == "__main__":
    print("Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
