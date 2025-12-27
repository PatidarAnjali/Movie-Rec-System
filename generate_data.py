import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("Generating movie dataset...")
np.random.seed(42)

genres = ['Action', 'Comedy', 'Drama', 'Sci Fi', 'Horror', 'Romance', 'Thriller']
movies = []

for i in range(100):
    movies.append({
        'movie_id': i,
        'title': f'Movie_{i}',
        'genre': np.random.choice(genres),
        'year': np.random.randint(2015, 2024),
        'avg_rating': round(np.random.uniform(2.5, 5.0), 1)
    })
movies_df = pd.DataFrame(movies)

users = []
for i in range(50):
    users.append({
        'user_id': i, 
        'age': np.random.randint(18, 65),
         'favorite_genre': np.random.choice(genres),
    })

users_df = pd.DataFrame(users)

ratings = []

for user_id in range(50):

    num_ratings = np.random.randint(10, 31)
    rated_movies = np.random.choice(100, num_ratings , replace = False)
    user_fav = users_df[users_df['user_id'] == user_id]['favorite_genre'].values[0]
    
    for movie_id in rated_movies:
        movie_genre = movies_df[movies_df['movie_id'] == movie_id]['genre'].values[0]
        if movie_genre == user_fav:
            rating = np.random.randint(3, 6)
        else:
            rating = np.random.randint(1, 6)
        ratings.append({
            'user_id': user_id,
            'movie_id': movie_id,
            'rating': rating,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
        })
ratings_df = pd.DataFrame(ratings)

movies_df.to_csv('movies.csv', index=False)
users_df.to_csv('users.csv', index=False)
ratings_df.to_csv('ratings.csv', index=False)

print("Dataset created successfully!")
print(f"   Movies: {len(movies_df)}")
print(f"   Users: {len(users_df)}")
print(f"   Ratings: {len(ratings_df)}")
