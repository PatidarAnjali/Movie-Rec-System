### Recommendation Model

An AI/ML recommendation system built w/ Python, featuring multiple recommendation algorithms & a FastAPI backend.

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
1. Install dependencises
```
# create virtual environment
python -m venv venv

# activate virtual environment 
# on Windows:
venv\Scripts\activate
# on macOS/Linux:
source venv/bin/activate

# install packages
pip install -r requirements.txt
```

2. Generate sample data
```
python rec_sys_data.py
<!-- 
This creates three CSV files:
    - movies.csv - 100 movies with genres, years, ratings
    - users.csv - 50 users with age and preferences
    - ratings.csv - ~1000 user-movie ratings 
-->
```

3. Train and Test Models
```
python rec_sys_models.py
<!--
This will:
- Train all four models
- Generate recommendations for a test user
- Display results in console
-->
```

4. Evaluate Models
```
python rec_sys_evaluation.py
<!-- 
This will:
- Split data into train/test sets
- Evaluate all models
- Generate comparison metrics
- Create visualization plot: (evaluation_results.png)
-->
```

5. Start API Server
```
python rec_sys_api.py

# Or using uvicorn directly:
bashuvicorn rec_sys_api:app --reload --host 0.0.0.0 --port 8000
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
  "model_type": "hybrid"
}
```

Get All Movies
```
GET /movies?genre=Action&limit=20
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
bashGET /stats
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

## Project Structure
```
recommendation-system/
├── rec_sys_data.py          # Data generation
├── rec_sys_models.py        # ML models implementation
├── rec_sys_evaluation.py    # Model evaluation
├── rec_sys_api.py          # FastAPI service
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── movies.csv              # Generated data
├── users.csv               # Generated data
├── ratings.csv             # Generated data
└── evaluation_results.png  # Generated plots
```

## Screenshots
<img width="2788" height="1350" alt="image" src="https://github.com/user-attachments/assets/e5a9e0eb-bf0c-479f-a543-3e589e241355" />
<img width="2624" height="1426" alt="image" src="https://github.com/user-attachments/assets/2168b8eb-4090-4e6b-8fcd-2a14cbd4183f" />
<img width="2642" height="1092" alt="image" src="https://github.com/user-attachments/assets/09ce833e-ec2f-4f6c-8356-275884406439" />


