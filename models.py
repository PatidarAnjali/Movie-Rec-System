import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

class SimpleRecommender:
    
    def __init__(self, ratings_df):
        self.ratings_df = ratings_df
        self.user_item_matrix = None
        self.item_similarity = None
        self._build()
    
    def _build(self):
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)

        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
    
    def recommend(self, user_id, n=10):

        if user_id not in self.user_item_matrix.index:
            return []
        
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        scores = {}

        for movie_id in self.user_item_matrix.columns:
            if movie_id not in rated_movies:
                movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                score = 0
                for rated_movie in rated_movies:
                    rated_idx = list(self.user_item_matrix.columns).index(rated_movie)
                    similarity = self.item_similarity[movie_idx][rated_idx]
                    score += similarity * user_ratings[rated_movie]
                scores[movie_id] = score
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [movie_id for movie_id, _ in recommendations[:n]]

class MatrixFactorization:
    def __init__(self, n_factors = 20):
        self.n_factors = n_factors
        self.model = TruncatedSVD(n_components=n_factors, random_state=42)
        self.user_factors = None
        self.item_factors = None
        self.user_item_matrix = None
    
    def fit(self, ratings_df):
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', columns='movie_id', values='rating'
        ).fillna(0)
        self.user_factors = self.model.fit_transform(self.user_item_matrix)
        self.item_factors = self.model.components_.T
    
    def recommend(self, user_id, n = 10):

        if user_id not in self.user_item_matrix.index:
            return []
        
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_ratings = self.user_item_matrix.loc[user_id]
        predictions = []

        for movie_id in self.user_item_matrix.columns:
            if user_ratings[movie_id] == 0:
                movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                pred = np.dot(self.user_factors[user_idx], self.item_factors[movie_idx])
                predictions.append((movie_id, pred))
        predictions.sort(key=lambda x: x[1], reverse = True)

        return [movie_id for movie_id, _ in predictions[:n]]

if __name__ == "__main__":
    print("Loading data...")
    ratings_df = pd.read_csv('ratings.csv')
    movies_df = pd.read_csv('movies.csv')
    print("Training models...")
    simple = SimpleRecommender(ratings_df)
    mf = MatrixFactorization(n_factors=20)
    mf.fit(ratings_df)
    test_user = 5
    print(f"\nRecommendations for User {test_user}:")
    recs = simple.recommend(test_user, n=5)
    for i, movie_id in enumerate(recs, 1):
        movie = movies_df[movies_df['movie_id'] == movie_id].iloc[0]
        print(f"  {i}. {movie['title']} ({movie['genre']}, {movie['year']})")
    print("\nModels working!")
