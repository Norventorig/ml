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

movies = user_tags['movieId'].unique()
tags = [' | '.join(user_tags[user_tags['movieId'] == i_id]['tag'].
                   apply(lambda x: str(x).lower()).unique())
        for i_id in movies]
ratings = [user_ratings[user_ratings['movieId'] == i_id]['rating'].values[0] for i_id in movies]
genres = [user_genres[user_genres['movieId'] == i_id]['genres'].values[0] for i_id in movies]

data = {'movieId': movies, 'genres': genres, 'tags': tags, 'rating': ratings}
df = p.DataFrame(data=data)

print(f"Число уникальных тэгов: {len(set(df['tags'].str.cat(sep=' | ').split(' | ')))}")

tf_idf = TfidfVectorizer(min_df=5, max_df=0.8)
