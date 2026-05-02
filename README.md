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
- Collaborative Filtering (item similarity)
- Matrix Factorization (SVD via TruncatedSVD)
- RESTful API with Swagger docs
- Real-time recommendations

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
├── api.py                  # FastAPI server w/ recommendation endpoints
├── models.py               # Recommenders (item-similarity + matrix factorization)
├── generate_data.py        # Synthetic dataset generator
├── test_system.py          # Testing script for validation
├── rec_sys_api.py          # Convenience runner (uvicorn -> api:app)
├── rec_sys_models.py       # Convenience runner for model smoke test
├── rec_sys_data.py         # Convenience runner for data generation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── movies.csv              # Generated movie data
├── users.csv               # Generated user data
└── ratings.csv             # Generated ratings data
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

Get All Movies
```text
GET /movies?limit=20
```

Get User Details
```text
GET /users/5
```

Add Rating
```json
POST /ratings
{
  "user_id": 5,
  "movie_id": 42,
  "rating": 4.5
}
```

System Stats
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

## Screenshot proof ideas
- Swagger UI running: `http://localhost:8000/docs`
- Example API response: run the cURL recommend call and screenshot the JSON output
- Terminal proof: screenshot `python models.py` (or `python rec_sys_models.py`) output

### File Descriptions
- **`api.py`**: FastAPI REST API server with endpoints for recommendations, movies, and health checks
- **`models.py`**: Implementation of recommendation algorithms
  - `SimpleRecommender`: Item-based collaborative filtering
  - `MatrixFactorization`: SVD-based factorization model
- **`generate_data.py`**: Creates synthetic movie dataset with realistic rating patterns
- **`test_system.py`**: Automated testing for data, models, and API
- **`requirements.txt`**: All Python package dependencies

## Screenshots
<img width="2788" height="1350" alt="image" src="https://github.com/user-attachments/assets/e5a9e0eb-bf0c-479f-a543-3e589e241355" />
<img width="2624" height="1426" alt="image" src="https://github.com/user-attachments/assets/2168b8eb-bf0c-479f-a543-3e589e241355" />
<img width="2642" height="1092" alt="image" src="https://github.com/user-attachments/assets/09ce833e-ec2f-4f6c-8356-275884406439" />

