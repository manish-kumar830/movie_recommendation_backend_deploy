from flask import Flask, jsonify, request
import pandas as pd
import requests
import pickle
import os
import numpy as np

app = Flask(__name__)

# Enable CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

df_url  = "https://model98989.000webhostapp.com/model/movies_dataframe.pkl"
model_url  = "https://model98989.000webhostapp.com/model/movies_dataframe.pkl"


with open(os.path.join(MODEL_DIR,'movies_dataframe.pkl'),'wb') as f:
   f.write(requests.get(df_url).content)


with open(os.path.join(MODEL_DIR,'movies_similarity_model.pkl'),'wb') as f:
   f.write(requests.get(model_url).content)

# Load movies dataframe and similarity model
movies_df = pd.DataFrame(pickle.load(open(os.path.join(MODEL_DIR, 'movies_dataframe.pkl'), "rb")))
movie_similarity = pickle.load(open(os.path.join(MODEL_DIR, 'movies_similarity_model.pkl'), "rb"))


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8e4f3f6efc94cce5b18194960d78de95&language=en-US".format(
        movie_id)
    data = requests.get(url).json()
    poster_path = data.get('poster_path', None)
    if poster_path is None:
        return "https://www.filmfodder.com/reviews/images/poster-not-available.jpg"
    else:
        return "https://image.tmdb.org/t/p/w500" + poster_path


@app.route("/")
def index():
    movies_df.head()
    return jsonify({"message": "Welcome To Movie Recommendation Api"})


@app.route("/movie_title", methods=['GET'])
def title():
    x = movies_df['original_title'].values.tolist()
    l = [{"value": i, "label": i} for i in x]
    return jsonify({"title": l})


@app.route("/recommend-movie/<movie_title>", methods=['GET'])
def recommend_movie(movie_title):
    movie_index = movies_df[movies_df['original_title'] == movie_title].index[0]
    similar_movies_list = sorted(list(enumerate(movie_similarity[movie_index])), reverse=True,
                                 key=lambda x: x[1])[1:6]

    movie_list = []
    for i in similar_movies_list:
        movies_dict = {
            "imdb_id": movies_df.iloc[i[0]].imdb_id,
            "movie_name": movies_df.iloc[i[0]].original_title,
            "poster_path": fetch_poster(movies_df.iloc[i[0]].imdb_id)
        }
        movie_list.append(movies_dict)

    return jsonify({"recommendation": movie_list})


@app.route("/movie_title_poster_path", methods=['GET'])
def titlePath():
    movie_object_list = [{"original_title": movies_df['original_title'][i],
                          "imdb_id": movies_df['imdb_id'][i],
                          "poster_path": fetch_poster(movies_df['imdb_id'][i])} for i in range(28)]

    return jsonify({"movie_list": movie_object_list})


# if __name__ == "__main__":
#    #  port = int(os.getenv("PORT", 3000))
#    #  app.run(host="0.0.0.0", port=port, debug=True)
#    app.run()
