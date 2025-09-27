import pandas as p
import numpy as n

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


links = p.read_csv('links.csv')
movies = p.read_csv('movies.csv')
ratings = p.read_csv('ratings.csv')
tags = p.read_csv('tags.csv')

model = LinearRegression()

user_id = ratings['userId'].value_counts().index[2]
user_ratings = ratings[ratings['userId'] == user_id][['rating', 'movieId']]

user_tags = tags[tags['movieId'].isin(user_ratings['movieId'].to_list())][['tag', 'movieId']]
user_genres = movies[movies['movieId'].isin(user_tags['movieId'].to_list())][['genres', 'movieId']]
user_ratings = user_ratings[user_ratings['movieId'].isin(user_genres['movieId'].to_list())]

print(f"Максимальное число жанров на фильме: {user_genres['genres'].apply(lambda x: len(x.split('|'))).max()}")

# counts = user_tags['movieId'].value_counts()
# user_tags = user_tags[user_tags['movieId'].isin(counts[counts <= 3].index)]

# | Это не то ты должен сделать вектор по тэгам следовательно сократить числ самих уникальных тегов