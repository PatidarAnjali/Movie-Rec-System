### Recommendation Model

An AI/ML movie recommendation system built with Python, featuring a FastAPI backend and multiple recommendation approaches (item similarity + matrix factorization).

## Features
- Multiple recommendation algorithms
    - collaborative filtering (user based & item based)
    - matrix factorization (SVD)
    - content based filtering
    - hybrid model (combines all approaches)
- RESTful API
    - FastAPI based service for easy integration 
- Comprehensive evaluation
    - RMSE, MAE, Precision, Recall, NDCG metrics
- Production ready
    - Docker support
    - API documentation
    - caching

## Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended) 

## Quick start
1. Install dependencies
```
# create virtual environment
python3 -m venv .venv

# activate virtual environment 
# on Windows:
.venv\Scripts\activate
# on macOS/Linux:
source .venv/bin/activate

# install packages
pip install -r requirements.txt
```

2. Generate sample data
```
python rec_sys_data.py
```

3. Train and Test Models
```
python rec_sys_models.py
```

4. Start API Server
```
python rec_sys_api.py

# Or using uvicorn directly:
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

API will be available at: http://localhost:8000

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Key Endpoints
Get Recommendations
```
POST /recommend
{
  "user_id": 5,
  "n": 10,
}
```

Get All Movies
```
GET /movies?limit=20
```

Get User Details
```
GET /users/5
```

Add Rating
```
POST /ratings
{
  "user_id": 5,
  "movie_id": 42,
  "rating": 4.5
}
```

System Stats
```
GET /stats
```

## Testing with cURL
```
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 5, "n": 5, "model_type": "hybrid"}'

# Get movies
curl "http://localhost:8000/movies?genre=Action&limit=10"

# Health check
curl "http://localhost:8000/health"
```

## Screenshot proof ideas
- Swagger UI running: `http://localhost:8000/docs`
- Example API response: run the cURL recommend call and screenshot the JSON output
- Terminal proof: screenshot `python rec_sys_models.py` output

Project Structure
```
recommendation-system/
├── rec_sys_data.py          # Data generation
├── rec_sys_models.py        # ML models implementation
├── rec_sys_api.py          # FastAPI service
├── generate_data.py         # Data generator (implementation)
├── models.py                # Recommenders (implementation)
├── api.py                   # FastAPI app (implementation)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── movies.csv              # Generated data
├── users.csv               # Generated data
├── ratings.csv             # Generated data
```

