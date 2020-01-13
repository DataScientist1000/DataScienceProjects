# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os

path = os.path.abspath("Recommendation/Data/movies.csv")

rpath = os.path.abspath("Recommendation/Data/ratings.csv")

"""
#EDA
data['title'] = data['title'].apply('str')
data['title'] = data['title'].astype(basestring)

"""
moviedata = pd.read_csv(path, usecols = ['movieId','title'], dtype = {'movieId': 'int64', 'title':'str'})

ratingdata = pd.read_csv(rpath, usecols = ['userId','movieId','rating'], dtype = {'userId':'int64','movieId':'int64','rating':'float32'})

moviedata.info()
ratingdata.info()

moviedata.columns
moviedata.movieId.dtypes

moviedata['title'] = moviedata['title'].astype(str)

moviedata.head()
ratingdata.head()

#MErge the df

df = pd.merge(moviedata,ratingdata,how ='inner',on ='movieId')

df.head()

comb_movie_rating = df.dropna(axis=0, subset=['title'])

#there is no change in df and comb_movie_rating it acts as just copy

movie_rating_count = (comb_movie_rating.groupby(by=['title'])['rating'].
                      count().reset_index().
                      rename(columns = {'rating':'toatlRating'})
                      [['title','toatlRating']])





