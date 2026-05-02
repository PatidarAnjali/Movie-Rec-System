from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import uvicorn
from models import SimpleRecommender

app = FastAPI(title="Movie Recommendation API")
model = None
movies_df = None
users_df = None
ratings_df = None

class RecommendRequest(BaseModel):
    user_id: int
    n: Optional[int] = 10


class RatingRequest(BaseModel):
    user_id: int
    movie_id: int
    rating: float

@app.on_event("startup")
async def load_data():
    global model, movies_df, users_df, ratings_df
    print("Loading data...")
    movies_df = pd.read_csv('movies.csv')
    users_df = pd.read_csv("users.csv")
    ratings_df = pd.read_csv('ratings.csv')
    print("Training model...")
    model = SimpleRecommender(ratings_df)
    print("Model ready!")

@app.get("/")
async def root():
    return {"message": "Movie Recommendation API", "docs": "/docs"}

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/stats")
async def stats():
    if movies_df is None or users_df is None or ratings_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    return {
        "movies": int(len(movies_df)),
        "users": int(len(users_df)),
        "ratings": int(len(ratings_df)),
    }

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

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if users_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")
    user = users_df[users_df["user_id"] == user_id]
    if user.empty:
        raise HTTPException(status_code=404, detail="User not found")
    return user.iloc[0].to_dict()

@app.post("/ratings")
async def add_rating(request: RatingRequest):
    global ratings_df, model
    if ratings_df is None or movies_df is None or users_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded")

    if users_df[users_df["user_id"] == request.user_id].empty:
        raise HTTPException(status_code=404, detail="User not found")
    if movies_df[movies_df["movie_id"] == request.movie_id].empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    if not (0.0 <= request.rating <= 5.0):
        raise HTTPException(status_code=400, detail="Rating must be between 0 and 5")

    new_row = pd.DataFrame(
        [{"user_id": request.user_id, "movie_id": request.movie_id, "rating": request.rating}]
    )
    ratings_df = pd.concat([ratings_df, new_row], ignore_index=True)
    ratings_df.to_csv("ratings.csv", index=False)

    model = SimpleRecommender(ratings_df)
    return {"ok": True}

if __name__ == "__main__":
    print("Starting API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
