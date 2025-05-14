from utils.util import read_dat
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import pandas as pd
import os

print(os.path.abspath("."))
base_dir = "./ml-1m/"
movie_path = "movies.dat"
rating_path = "ratings.dat"
users_path = "users.dat"


def compute_rmse(actual_matrix, predicted_matrix):
    actual = actual_matrix.values.flatten()
    predicted = predicted_matrix.values.flatten()

    mask = ~np.isnan(actual)

    rmse = sqrt(mean_squared_error(actual[mask], predicted[mask]))
    return rmse


def train(ratings_matrix, ratings_matrix_filled, n):
    nmf = NMF(n_components=n, init="nndsvd", random_state=42, max_iter=500)
    W = nmf.fit_transform(ratings_matrix_filled)
    H = nmf.components_
    ratings_pred = np.dot(W, H)
    ratings_pred_df = pd.DataFrame(
        ratings_pred, index=ratings_matrix.index, columns=ratings_matrix.columns
    )

    return ratings_pred, ratings_pred_df


def test_n_component(ratings_matrix, ratings_matrix_filled):
    component_range = [5, 15, 25, 40, 55, 110]
    results = []

    for n in component_range:
        _, ratings_pred_df = train(ratings_matrix, ratings_matrix_filled, n)
        rmse = compute_rmse(ratings_matrix, ratings_pred_df)
        results.append((n, rmse))

    plt.figure(figsize=(8, 5))
    plt.plot([x[0] for x in results], [x[1] for x in results], marker="o")
    plt.title("RMSE vs Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("RMSE")
    plt.xticks(component_range)
    plt.show()


def recommend_movie(user_index, rating_matrix, ratings_pred, n=2):
    user_id = rating_matrix.index[user_index]

    unwatch_movies = rating_matrix.loc[user_id][
        rating_matrix.loc[user_id].isna()
    ].index.tolist()

    pred_ratings = pd.Series(ratings_pred[user_index], index=rating_matrix.columns)
    top_recommendations = pred_ratings.loc[unwatch_movies].nlargest(n)
    return top_recommendations.to_dict()


def main():
    movie_data: pd.DataFrame = read_dat(
        os.path.join(base_dir) + movie_path,
        ";",
        ["movie_id", "moviename", "genre"],
    )

    rating_data: pd.DataFrame = read_dat(
        os.path.join(base_dir) + rating_path,
        ";",
        ["user_id", "movie_id", "rating", "timestamp"],
    )

    pivot = rating_data.pivot(columns="movie_id", index="user_id", values="rating")
    pivot_filled = pivot.fillna(0)

    test_n_component(pivot, pivot_filled)
    ratings_pred, _ = train(pivot, pivot_filled, 55)

    user_index = 66
    top_recommendations = recommend_movie(user_index, pivot, ratings_pred)

    for movie_id, rating in top_recommendations.items():
        moviename = movie_data.loc[movie_data["movie_id"] == movie_id][
            "moviename"
        ].values[0]
        print(f"{moviename} - Rating: {rating:.2f} for User {user_index}")


if __name__ == "__main__":
    main()
