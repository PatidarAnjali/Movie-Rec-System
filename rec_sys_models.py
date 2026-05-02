from models import MatrixFactorization, SimpleRecommender
import pandas as pd


def main() -> None:
    print("Loading data...")
    ratings_df = pd.read_csv("ratings.csv")
    movies_df = pd.read_csv("movies.csv")

    print("Training models...")
    simple = SimpleRecommender(ratings_df)
    mf = MatrixFactorization(n_factors=20)
    mf.fit(ratings_df)

    test_user = 5
    print(f"\nRecommendations for User {test_user}:")
    recs = simple.recommend(test_user, n=5)
    for i, movie_id in enumerate(recs, 1):
        movie = movies_df[movies_df["movie_id"] == movie_id].iloc[0]
        print(f"  {i}. {movie['title']} ({movie['genre']}, {movie['year']})")

    print("\nModels working!")


if __name__ == "__main__":
    main()
