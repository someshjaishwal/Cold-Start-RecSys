import numpy as np
import pandas as pd
import preprocess
import algo
import recommend
import time

# raw data
movies_raw = pd.read_table('./ml-1m/movies.dat', sep = '::', names = ['MovieID', 'Title', 'Genres'], engine = 'python')
ratings_raw = pd.read_table('./ml-1m/ratings.dat', sep = '::', names = ['UserID', 'MovieID', 'Rating', 'Timestamp'],engine = 'python')
users_raw = pd.read_table('./ml-1m/users.dat', sep = '::', names = ['UserID','Gender','Age','Occupation','Zip-code'], engine = 'python')

###################################### Part - 1 Refining dataset and creating test set

# separting test sets i.e. new users and new movies
'''
ratings_raw, user_test, movie_test = preprocess.split(ratings_raw, fraction = 0.1)

ratings_raw.to_csv('ratings_raw.csv', sep = '\t',index=False)
user_test.to_csv('user_test.csv', sep = '\t',index=False)
movie_test.to_csv('movie_test.csv', sep = '\t',index=False)
'''
ratings_raw = pd.read_csv('ratings_raw.csv', sep = '\t')
user_test = pd.read_csv('user_test.csv', sep = '\t')
movie_test = pd.read_csv('movie_test.csv', sep = '\t')

# refining ratings_raw
ratings_nan = ratings_raw.iloc[:,0:4]
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

###################################### PART - 2  Probabilistic Inductive Matrix Factorization
num_latent_features = [50]

# sgd
u, theta, W_u, v, phi, W_v = algo.prob_mf_sgd(ratings_raw, N = ratings_nan.shape[0], M = ratings_nan.shape[1], 
                                              K = 10 , users = users.iloc[:,1:], movies = movies,
                                              momentum = 0.8,
                                              learning_rate = 0.05,
                                              lambda_u = 0.05,
                                              lambda_v = 0.05,
                                              lambda_x = 0.05,
                                              itr_count = 1000) #2000000


# alt opt
alt_opt_start_time = time.time()
for k in num_latent_features: 
    '''
    u, theta, W_u, v, phi, W_v, rmse = algo.mf_alt_opt(ratings_raw, N = ratings_nan.shape[0], M = ratings_nan.shape[1], 
                                                  K = k , users = users.iloc[:,1:], movies = movies,
                                                  lambda_u = 0.05,
                                                  lambda_v = 0.05,
                                                  lambda_x = 0.05,
                                                  itr_count = 20) #2000000
    np.savetxt('rmse_alt_opt_k_'+str(k)+'.txt',rmse)
    '''
    # clusterification
    # apply pca and visualize
    u_pca = algo.visualize_pca_features(u, k)
    u_tsne = algo.visualize_tsne_features(u, k)

alt_opt_time_taken = time.time() - alt_opt_start_time

########### Part 2 - sub part : clusterification

# dendograms
algo.show_dendogram(u_pca,K=50)
algo.show_dendogram(u_tsne,K=50)

# clusterify users
X = u_tsne
clusters = algo.get_clusters(X, num_clusters = 3)

# visualize clusters
algo.visualize_clusters(X,clusters, num_clusters = 3, K = 50)
# get classifier for latent taste of users vs clusters
classifier, cm = algo.get_classifier(X, clusters)
  
#################################### PART - 3  Recommending movies to new user

# latent features for new user
new_user = recommend.get_a_new_user(user_test,users_raw)
user_latent_features  = np.dot(new_user.values[:,1:], W_u).flatten()

# bias for new user
regressor_u , rmse_train, rmse_test = algo.ridge_regression(users.values[:,1:], theta , alpha = 1)
u_bias = regressor_u.predict(new_user.values[:,1:]).flatten()

# cluster of new user
his_cluster = classifier.predict(user_latent_features.reshape(-1,2))[0]

cluster_hits = recommend.cluster_hits(ratings_nan, clusters, movies_raw, his_cluster, n = 5)
recommended_movies = recommend.movies(user_latent_features, v, movies_raw, 20 , u_bias, phi)
wishlist_of_user = recommend.get_users_wishlist(new_user['UserID'], user_test, movies_raw)

recommended_in_wishlist = pd.merge(recommended_movies, wishlist_of_user, how='inner', on=['MovieID']).sort_values('Rating', ascending = False)
cluster_hits_in_recommended = pd.merge(recommended_movies, cluster_hits, how='inner', on=['MovieID'])
cluster_hits_in_wishlist = pd.merge(wishlist_of_user, cluster_hits, how='inner', on=['MovieID'])

###################### Saving Recommendations to new user into excel

writer = pd.ExcelWriter('rec_for_user_5.xlsx')

recommended_movies.to_excel(writer,'Recommended Movies')
wishlist_of_user.to_excel(writer,'User\'s Wishlist')
recommended_in_wishlist.to_excel(writer,'Recommended in Wishlist')

cluster_hits.to_excel(writer,'Popular in Cluster')
cluster_hits_in_recommended.to_excel(writer,'Cluster Popular in Recommend.')
cluster_hits_in_wishlist.to_excel(writer,'Cluster Popular in Wishlist.')
writer.save()

##################################### Part - 4 Recommnding users to new movie

# latent features for new movie
new_movie_id , new_movie = recommend.get_new_movie(movie_test, movies_raw, genre_mapping)
movie_latent_features = np.dot(new_movie, W_v)

# bias for new movie
regressor_v , rmse_train, rmse_test = algo.ridge_regression(movies, phi , alpha = 1)
v_bias = regressor_v.predict(new_movie.reshape(1,new_movie.shape[0])).flatten()

recommended_users = recommend.users(movie_latent_features, u, users_raw, 20 , v_bias, theta)
wishlist_of_movie = recommend.get_movies_wishlist(new_movie_id.values,movie_test,users_raw)

test_user_recommendations = pd.merge(recommended_users, wishlist_of_movie, how='inner', on=['UserID']).sort_values('Rating', ascending = False)

### save 
writer = pd.ExcelWriter('rec_for_movie_.xlsx')

recommended_users.to_excel(writer,'Recommended users')
wishlist_of_movie.to_excel(writer,'Movie\'s Promo Site list')
test_user_recommendations.to_excel(writer,'Recommended in Site list')

writer.save()