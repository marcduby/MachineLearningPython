

# imports
# pip3 install numpy pandas matplotlib scikit-learn faiss-cpu openai sentence-transformers
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
# from openai import OpenAI
from matplotlib import pyplot as plt
import os


# constants
DIR_DATA = "/Users/mduby/Code/PythonWorkspace/MachineLearningPython/PyData/Workshops/PydataBoston2025/movies-ml-latest-small/{}"
FILE_MOVIES = DIR_DATA.format("movies.csv")
FILE_RATINGS = DIR_DATA.format("ratings.csv")


# methods
def test():
    '''
    test method
    '''
    print("dude\n\n\n")


def load_data():
    '''
    Docstring for load_data
    '''
    # TODO: Load the movies dataset
    movies = pd.read_csv(FILE_MOVIES)

    # TODO: Load the ratings dataset
    ratings = pd.read_csv(FILE_RATINGS)

    # TODO: Print number of rows in each DataFrame
    print(f"Loaded {len(movies)} movies and {len(ratings)} ratings")

    #return
    return movies, ratings


def clean_movies(df_movies):
    '''
    Docstring for clean_movies
    
    :param movies: Description
    '''
    items = df_movies.copy()

    # TODO: Replace genre separators ('|') with commas for readability
    items['genres'] = df_movies["genres"].str.replace("|", ",", regex=False)

    # 2️⃣ Rename columns
    df_items = df_movies.rename(columns={"movieId": "id", "genres": "tags"})

    # TODO (Optional): Display the first few rows to verify your transformations
    print(df_items.head())

    # return
    return df_items


# main
if __name__ == "__main__":
    df_movies, df_ratings = load_data()

    df_movies = clean_movies(df_movies=df_movies)

    print(f"Loaded {len(df_movies)} movies and {len(df_ratings)} ratings")
