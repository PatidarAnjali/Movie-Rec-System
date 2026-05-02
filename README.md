### Recommendation Model

An AI/ML movie recommendation system built with Python, featuring a FastAPI backend and multiple recommendation approaches (item similarity + matrix factorization).

## Tech Stack
- Python 3.9+
- FastAPI (REST API framework)
- scikit-learn (ML algorithms)
- Pandas & NumPy (data processing)

## Quick start
```bash
# create virtual environment
python3 -m venv .venv

# activate virtual environment
# on Windows:
.venv\Scripts\activate
# on macOS/Linux:
source .venv/bin/activate

# install packages
pip install -r requirements.txt

# Generate sample data
python3 generate_data.py
# (or) python3 rec_sys_data.py

# Test models
python3 models.py
# (or) python3 rec_sys_models.py

# Start API server
python3 api.py
# (or) python3 rec_sys_api.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## Features
- collaborative filtering (item similarity)
- matrix factorization (SVD via TruncatedSVD)
- RESTful API w/ swagger docs
- real-time recommendations

## API Endpoints
- `POST /recommend` - get personalized recommendations
- `GET /movies` - browse movies
- `GET /users/{user_id}` - get a user
- `POST /ratings` - add a rating
- `GET /stats` - dataset stats
- `GET /health` - health check

## Algorithms
1. Collaborative Filtering: item-item similarity using cosine similarity
2. Matrix Factorization: latent factor model using truncated SVD

## Project Structure
```text
recommendation-model/
├── api.py # FastAPI server w/ recommendation endpoints
├── models.py # recommenders (item-similarity + matrix factorization)
├── generate_data.py # synthetic dataset generator
├── test_system.py # testing script for validation
├── rec_sys_api.py # convenience runner (uvicorn -> api:app)
├── rec_sys_models.py # convenience runner for model smoke test
├── rec_sys_data.py # convenience runner for data generation
├── requirements.txt # python dependencies
├── README.md # project documentation
├── .gitignore # git ignore rules
│
├── movies.csv #generated movie data
├── users.csv # generated user data
└── ratings.csv # generated ratings data
```

## Key Endpoints
Get Recommendations
```json
POST /recommend
{
  "user_id": 5,
  "n": 10
}
```

Get All movies
```text
GET /movies?limit=20
```

Get user details
```text
GET /users/5
```

Add rating
```json
POST /ratings
{
  "user_id": 5,
  "movie_id": 42,
  "rating": 4.5
}
```

System stats
```text
GET /stats
```

## Testing with cURL
```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 5, "n": 5}'

# Get movies
curl "http://localhost:8000/movies?limit=10"

# Health check
curl "http://localhost:8000/health"
```

## Screenshot
- swagger UI running: `http://localhost:8000/docs`
<img width="1153" height="744" alt="Screenshot 2026-05-02 at 4 41 47 PM" src="https://github.com/user-attachments/assets/5a8012ee-79c5-4670-9349-c258b16b66c9" />
<img width="729" height="266" alt="Screenshot 2026-05-02 at 4 42 23 PM" src="https://github.com/user-attachments/assets/80b1df6d-8ef9-44e3-850d-2c650bd6599c" />

### File Descriptions
- **`api.py`**: FastAPI REST API server with endpoints for recommendations, movies, and health checks
- **`models.py`**: implementation of recommendation algorithms
  - `SimpleRecommender`: item-based collaborative filtering
  - `MatrixFactorization`: SVD-based factorization model
- **`generate_data.py`**: creates synthetic movie dataset with realistic rating patterns
- **`test_system.py`**: automated testing for data, models, and API
- **`requirements.txt`**: all python package dependencies

