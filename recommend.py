import numpy as np
import pandas as pd
from random import randint
import preprocess

def movies(user_latent_features, v, movies_raw , num_movies, theta , phi):
    preferences = np.dot(v,user_latent_features) + phi + theta
    preferences = preferences.argsort()[-num_movies:][::-1] + 1
    preferences = pd.DataFrame(preferences,columns=['MovieID'])
    recommendations = preferences.merge(movies_raw, how = 'left', left_on = 'MovieID', right_on = 'MovieID')
    return recommendations #dataframe

def users(movie_latent_features, u, users_raw, num_users, phi, theta):
    users = np.dot(u, movie_latent_features) + theta + phi
    users = users.argsort()[-num_users:][::-1] + 1
    users = pd.DataFrame(users,columns=['UserID'])
    recommended_users = users.merge(users_raw, how = 'left', left_on = 'UserID',  right_on = 'UserID')
    return recommended_users #dataframe

def get_a_new_user(user_test,users_raw):
    random_row = randint(1,user_test.shape[0]) - 1
    x = user_test.iloc[random_row].to_frame()
    x.columns = ['values']  
    x['ones'] = 1
    x = x.reset_index().pivot(index='ones', columns='index', values = 'values').drop(["MovieID","Rating"], axis = 1)
    x['UserID']=x['UserID'].astype('int64')
    x = users_raw.merge(x, how = 'right', left_on = 'UserID', right_on = 'UserID')
    new_user = preprocess.process_user_info(x)

    return new_user

def get_users_wishlist(UserID, user_test, movies_raw):
    UserID = UserID.astype('int64')
    wishlist_id = user_test.loc[user_test['UserID'] == UserID[0]].drop(columns = ['UserID'])
    wishlist_id["MovieID"] = wishlist_id["MovieID"].astype('int64')
    wishlist = movies_raw.merge(wishlist_id, how = 'right', left_on = 'MovieID', right_on = 'MovieID').sort_values('Rating', ascending = False)
    
    return wishlist
    
def get_new_movie(movie_test, movies_raw , genre_mapping):
    
    random_row = randint(1,movie_test.shape[0]) - 1
    x = movie_test.iloc[random_row].to_frame()
    x.columns = ['values'] 
    x['ones'] = 1
    x = x.reset_index().pivot(index='ones', columns='index', values = 'values').drop(["UserID","Rating"], axis = 1)
    x['MovieID']=x['MovieID'].astype('int64')
    movie_id = x['MovieID']
    x = movies_raw.merge(x, how = 'right', left_on = 'MovieID', right_on = 'MovieID').values[0,:].tolist()
    
    return movie_id, preprocess.new_movie_info(x, genre_mapping)

def get_movies_wishlist(new_movie_id,movie_test,users_raw):
    new_movie_id = new_movie_id.astype('int64')
    wishlist_id = movie_test.loc[movie_test['MovieID'] == new_movie_id[0]].drop(columns = ['MovieID'])
    wishlist_id["UserID"] = wishlist_id["UserID"].astype('int64')
    wishlist = users_raw.merge(wishlist_id, how = 'right', left_on = 'UserID', right_on = 'UserID').sort_values('Rating', ascending = False)
    
    return wishlist

def cluster_hits(ratings_nan, clusters, movies_raw, C, n):
    ratings_nan.fillna(0, inplace=True)
    ratings = ratings_nan.values[clusters == C]
    local_rating = np.sum(ratings,axis = 0)
    top_n = local_rating.argsort()[-n:][::-1] + 1
    top_n = pd.DataFrame(top_n,columns=['MovieID'])
    local_recommendations = top_n.merge(movies_raw, how = 'left', left_on = 'MovieID', right_on = 'MovieID')    
    return local_recommendations
    
    

