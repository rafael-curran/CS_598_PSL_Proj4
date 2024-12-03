import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Input, Output, callback_context
import streamlit as st
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State

# Fetch movie data from the GitHub URL
movies_url = "https://liangfgithub.github.io/MovieData/movies.dat?raw=true"
image_url_base = "https://liangfgithub.github.io/MovieImages/"

# Load movies data
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

# Load the similarity matrix
S_df = pd.read_csv("S_df.csv", index_col=0)

# Filter movies to include only those present in S_df
movies["FormattedMovieID"] = movies["MovieID"].apply(lambda x: f"m{x}")
filtered_movies = movies[movies["FormattedMovieID"].isin(S_df.index)]

# Select a subset of movies for display
sample_movies = filtered_movies.head(120)  # Adjust number of movies as needed

# Define the myIBCF function
def myIBCF(newuser):
    """
    Predicts ratings for movies not rated by a new user using Item-Based Collaborative Filtering (IBCF).
    """
    # Ensure the input user vector is a NumPy array
    w = np.asarray(newuser).flatten()

    # Convert S_df back to a NumPy array for computation
    S = S_df.to_numpy()
    
    # Validate dimensions
    if w.shape[0] != S.shape[0]:
        raise ValueError(f"Input vector length {w.shape[0]} does not match the similarity matrix dimensions {S.shape[0]}.")

    # Initialize predictions vector
    predictions = np.full_like(w, np.nan, dtype=float)

    # Compute predictions for each movie i
    for i in range(S.shape[0]):
        if not np.isnan(w[i]):  # Skip movies already rated by the user
            continue
        
        # Find movies rated by the user that have a similarity score with movie i
        rated_movies = np.where(~np.isnan(w))[0]  # Indices of movies rated by the user
        similarity_scores = S[i, rated_movies]   # Similarity scores with rated movies

        # Filter for non-NaN similarities
        valid_indices = ~np.isnan(similarity_scores)
        rated_movies = rated_movies[valid_indices]
        similarity_scores = similarity_scores[valid_indices]

        # Skip if no valid similarities are found
        if len(similarity_scores) == 0:
            continue

        # Compute the prediction
        numerator = np.sum(similarity_scores * w[rated_movies])
        denominator = np.sum(similarity_scores)

        # Avoid division by zero
        if denominator > 0:
            predictions[i] = numerator / denominator

    # Get movie recommendations
    recommendations = pd.Series(predictions, index=S_df.index)
    top_predictions = recommendations.sort_values(ascending=False).dropna().head(10)

    return top_predictions.index.tolist()

# Streamlit Implementation
def streamlit_app():
    st.title("Movie Recommendation System")

    st.write("Rate the following movies:")
    user_ratings = {}

    # Display movies in a grid layout
    columns_per_row = 4  # Number of movies per row
    for i in range(0, len(sample_movies), columns_per_row):
        cols = st.columns(columns_per_row)
        for idx, col in enumerate(cols):
            movie_idx = i + idx
            if movie_idx < len(sample_movies):
                row = sample_movies.iloc[movie_idx]
                with col:
                    st.image(row["image_url"], width=150)
                    st.write(row["Title"])
                    user_ratings[row["MovieID"]] = st.slider(
                        f"Rate {row['Title']}",
                        min_value=0,
                        max_value=5,
                        step=1,
                        value=0,
                        key=f"rating_{row['MovieID']}"
                    )

    # Get recommendations on button click
    if st.button("Get Recommendations"):
        # Prepare user input as a vector
        newuser = np.full(S_df.shape[0], np.nan)
        for movie_id, rating in user_ratings.items():
            if rating > 0:
                formatted_movie_id = f"m{movie_id}"
                movie_index = S_df.index.get_loc(formatted_movie_id)
                newuser[movie_index] = rating

        recommended_movie_ids = myIBCF(newuser)
        recommended_movies = movies[movies["MovieID"].isin([int(mid[1:]) for mid in recommended_movie_ids])]

        st.subheader("Recommended Movies:")
        cols = st.columns(len(recommended_movies))
        for col, (_, row) in zip(cols, recommended_movies.iterrows()):
            with col:
                st.image(row["image_url"], width=150, caption=row["Title"])



# Dash Implementation
app = Dash(__name__)

import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

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
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "streamlit":
        streamlit_app()
    else:
        app.run_server(debug=True)
