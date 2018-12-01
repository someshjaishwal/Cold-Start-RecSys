import numpy as np
import pandas as pd
import algo
import preprocess
import recommend

# raw data
movies_raw = pd.read_table('./ml-1m/movies.dat', sep = '::', names = ['MovieID', 'Title', 'Genres'], engine = 'python')
ratings_raw = pd.read_table('./ml-1m/ratings.dat', sep = '::', names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],engine = 'python')
users_raw = pd.read_table('./ml-1m/users.dat', sep = '::', names = ['UserID','Gender','Age','Occupation','Zip-code'], engine = 'python')

##################################### Part - 1 Refining dataset and creating test set

# separting test sets i.e. new users and new movies
ratings_raw, user_test, movie_test = preprocess.split(ratings_raw, fraction = 0.1)

# refining ratings_raw
ratings_nan = ratings_raw.iloc[:,0:3]
ratings_nan = ratings_nan.pivot(index='UserID', columns='MovieID', values='Rating')

# process info of users who have rated 
users = preprocess.user_info(users_raw,ratings_raw)

# process info of movie been rated 
movies, genres = preprocess.movie_info(movies_raw, ratings_raw)

# genre mapping : helps to convert a new movie vector
lst = []
for i in range(0,len(genres)):
    lst.append((genres[i],i))
genre_mapping = dict(lst)


###################################### Part - 2 Matrix Factorization using SGD

u, v = algo.mf_sgd(ratings_raw, N = ratings_nan.shape[0], M = ratings_nan.shape[1], K=10,
                   contribute_per_step = 1,
                   learning_rate = 0.005,
                   lamba_u = 0.0005,
                   lamba_v = 0.0005,
                   itr_count = 10000000)

#pred_ratings = np.linalg.multi_dot([u,v.T])

# Evaluate MF
rmse = algo.evaluate_mf_sgd(ratings_raw, u, v, theta = np.zeros(u.shape[0]), phi = np.zeros(v.shape[0]))


###################################### Part - 3 Recommending movies to new user

# latent features for new user
regressor_u , rmse_train, rmse_test = algo.ridge_regression(users.values[:,1:], u , alpha = 1)
new_user = recommend.get_a_new_user(user_test,users_raw)
user_latent_features  = regressor_u.predict(new_user.values[:,1:]).flatten()

recommended_movies = recommend.movies(user_latent_features, v, movies_raw, num_movies = 20, theta = 0, phi = np.zeros(v.shape[0]))
wishlist_of_user = recommend.get_users_wishlist(new_user['UserID'], user_test, movies_raw)

test_movie_recommendation = pd.merge(recommended_movies, wishlist_of_user, how='inner', on=['MovieID']).sort_values('Rating', ascending = False)

##################################### Part - 4 Recommnding users to new movie

#latent features for new movie
regressor_v ,rmse_train , rmse_test  = algo.ridge_regression(movies.values[:,:], v, 1)
new_movie_id , new_movie = recommend.get_new_movie(movie_test, movies_raw, genre_mapping)
movie_latent_features = regressor_v.predict(new_movie.reshape(1,new_movie.shape[0])).flatten()

recommended_users = recommend.users(movie_latent_features, u, users_raw, num_users = 20, phi = 0, theta = np.zeros(u.shape[0]))
wishlist_of_movie = recommend.get_movies_wishlist(new_movie_id.values,movie_test,users_raw)

test_user_recommendation = pd.merge(recommended_users, wishlist_of_movie, how='inner', on=['UserID']).sort_values('Rating', ascending = False)
