### Recommendation Model

An AI/ML recommendation system built w/ Python, that has collaborative filtering, matrix factorization, and content-based filtering.


## Tech Stack
- Python 3.9+
- FastAPI (REST API framework)
- scikit-learn (ML algorithms)
- Pandas & NumPy (data processing)

## Quick start
```bash
# Install dependencies
pip install -r requirements.txt

# Generate sample data
python3 generate_data.py

# Test models
python3 models.py

# Start API server
python3 api.py
```
Visit `http://localhost:8000/docs` for interactive API documentation.

## Features
- Collaborative Filtering (User & Item-based)
- Matrix Factorization (SVD)
- Content-Based Filtering
- RESTful API with Swagger docs
- Real-time recommendations

## API Endpoints
- `POST /recommend` - get personalized recommendations
- `GET /movies` - browse movies
- `GET /health` - health check

## Algorithms

1. Collaborative Filtering: Finds similar users/items using cosine similarity
2. Matrix Factorization: SVD-based latent factor model
3. Content-Based: Recommends based on genre and features

## Project Structure
```
movie-recommendation-system/
├── api.py                  # FastAPI server w/ recommendation endpoints
├── models.py              # ML models (Collaborative Filtering, Matrix Factorization)
├── generate_data.py       # synthetic dataset generator
├── test_system.py         # testing script for validation
├── requirements.txt       # python dependencies
├── README.md             # project documentation
├── .gitignore            # Git ignore rules
│
├── movies.csv            # generated movie data (100 movies)
├── users.csv             # generated user data (50 users)
└── ratings.csv           # generated ratings data (~1000 ratings)
```

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
<img width="2624" height="1426" alt="image" src="https://github.com/user-attachments/assets/2168b8eb-4090-4e6b-8fcd-2a14cbd4183f" />
<img width="2642" height="1092" alt="image" src="https://github.com/user-attachments/assets/09ce833e-ec2f-4f6c-8356-275884406439" />


