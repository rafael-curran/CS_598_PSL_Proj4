import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State

import os
os.chdir("/home/rafaelcurran/mysite")


movies_url = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
image_url_base = "https://liangfgithub.github.io/MovieImages/"

movies = pd.read_csv(
    movies_url,
    sep="::",
    engine="python",
    header=None,
    names=["MovieID", "Title", "Genres"],
    encoding="latin1"
)
movies["MovieID"] = movies["MovieID"].astype(int)
movies["Title"] = movies["Title"].apply(lambda x: x.encode('latin1').decode('utf-8', errors='replace'))
movies["image_url"] = movies["MovieID"].apply(lambda x: f"{image_url_base}{x}.jpg?raw=true")

S_df = pd.read_csv("S_df.csv", index_col=0)

movies["FormattedMovieID"] = movies["MovieID"].apply(lambda x: f"m{x}")
filtered_movies = movies[movies["FormattedMovieID"].isin(S_df.index)]

sample_movies = filtered_movies.head(120) 

def myIBCF(newuser):
    """
    Predicts ratings for movies not rated by a new user using Item-Based Collaborative Filtering (IBCF).
    """
    w = np.asarray(newuser).flatten()

    S = S_df.to_numpy()

    if w.shape[0] != S.shape[0]:
        raise ValueError(f"Input vector length {w.shape[0]} does not match the similarity matrix dimensions {S.shape[0]}.")

    predictions = np.full_like(w, np.nan, dtype=float)

    for i in range(S.shape[0]):
        if not np.isnan(w[i]): 
            continue

        rated_movies = np.where(~np.isnan(w))[0]  
        similarity_scores = S[i, rated_movies]  

        valid_indices = ~np.isnan(similarity_scores)
        rated_movies = rated_movies[valid_indices]
        similarity_scores = similarity_scores[valid_indices]

        if len(similarity_scores) == 0:
            continue

        numerator = np.sum(similarity_scores * w[rated_movies])
        denominator = np.sum(similarity_scores)

        if denominator > 0:
            predictions[i] = numerator / denominator

    recommendations = pd.Series(predictions, index=S_df.index)
    top_predictions = recommendations.sort_values(ascending=False).dropna().head(10)

    return top_predictions.index.tolist()


app = Dash(__name__)


app.layout = html.Div(
    [
        # First Section: Rate Your Movies
        html.Div(
            [
                html.H2(
                    "Step 1: Rate as many movies as possible",
                    style={
                        "backgroundColor": "#17a2b8",
                        "color": "white",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                ),
                dbc.Card(
                    dbc.CardBody(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Img(
                                            src=row["image_url"],
                                            style={
                                                "width": "120px",
                                                "display": "block",
                                                "margin": "0 auto",
                                            },
                                        ),
                                        html.Label(
                                            row["Title"],
                                            style={
                                                "textAlign": "center",
                                                "display": "block",
                                                "marginTop": "5px",
                                                "fontSize": "12px",
                                            },
                                        ),
                                        dcc.Slider(
                                            id=f"slider_{row['MovieID']}",
                                            min=0,
                                            max=5,
                                            step=1,
                                            value=0,
                                            marks={i: str(i) for i in range(6)},
                                            tooltip={"placement": "bottom"},
                                        ),
                                    ],
                                    style={
                                        "margin": "10px",
                                        "width": "150px",
                                        "display": "inline-block",
                                        "textAlign": "center",
                                        "border": "1px solid #ddd",
                                        "padding": "10px",
                                        "borderRadius": "5px",
                                    },
                                )
                                for _, row in sample_movies.iterrows()
                            ],
                            style={
                                "display": "flex",
                                "flexWrap": "wrap",
                                "justifyContent": "center",
                                "height": "300px",
                                "overflowY": "scroll",
                            },
                        )
                    )
                ),
            ]
        ),

        # Second Section: Recommended for You
        html.Div(
            [
                html.H2(
                    "Step 2: Discover movies you might like",
                    style={
                        "backgroundColor": "#17a2b8",
                        "color": "white",
                        "padding": "10px",
                        "borderRadius": "5px",
                        "textAlign": "center",
                    },
                ),
                dbc.Card(
                    dbc.CardBody(
                        html.Div(
                            [
                                html.Button(
                                    "Click here to get your recommendations",
                                    id="submit_button",
                                    className="btn btn-warning",
                                    n_clicks=0,
                                    style={"marginBottom": "10px"},
                                ),
                                html.Div(
                                    id="recommendations_output",
                                    style={
                                        "height": "300px",
                                        "overflowY": "scroll",
                                        "padding": "10px",
                                        "border": "1px solid #ddd",
                                        "borderRadius": "5px",
                                    },
                                ),
                            ],
                            style={"padding": "10px"},
                        )
                    )
                ),
            ]
        ),
    ]
)


@app.callback(
    Output("recommendations_output", "children"),
    [Input("submit_button", "n_clicks")],
    [State(f"slider_{row['MovieID']}", "value") for _, row in sample_movies.iterrows()],
)
def update_recommendations(n_clicks, *ratings):
    if n_clicks > 0:
        newuser = np.full(S_df.shape[0], np.nan)
        for idx, rating in enumerate(ratings):
            if rating > 0:
                movie_id = sample_movies.iloc[idx]["MovieID"]
                formatted_movie_id = f"m{movie_id}"
                movie_index = S_df.index.get_loc(formatted_movie_id)
                newuser[movie_index] = rating

        recommended_movie_ids = myIBCF(newuser)
        recommended_movies = movies[movies["MovieID"].isin([int(mid[1:]) for mid in recommended_movie_ids])]

        return html.Div([
            html.Div(
                [
                    html.Img(
                        src=row["image_url"],
                        style={
                            "width": "120px",
                            "display": "block",
                            "margin": "0 auto",
                        },
                    ),
                    html.Label(
                        row["Title"],
                        style={
                            "textAlign": "center",
                            "display": "block",
                            "marginTop": "5px",
                            "fontSize": "12px",
                        },
                    ),
                ],
                style={
                    "margin": "10px",
                    "width": "150px",
                    "display": "inline-block",
                    "textAlign": "center",
                    "border": "1px solid #ddd",
                    "padding": "10px",
                    "borderRadius": "5px",
                },
            )
            for _, row in recommended_movies.iterrows()
        ])

    return html.Div()

if __name__ == "__main__":
    app.run_server(debug=True)