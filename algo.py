import numpy as np
import pandas as pd
from random import randint
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def prob_mf_sgd(ratings_raw, N, M, K, users, movies, momentum, learning_rate, lambda_u, lambda_v, lambda_x, itr_count):
    
    ratings = ratings_raw.values[:,0:3]
    
    # for finding correct index of a movie
    map_movie_index = get_movie_index_map(ratings_raw)
    map_user_index = get_user_index_map(ratings_raw)
    
    # initialize matrices
    
    u = 0.1 * np.random.uniform(0,1, size=(N, K))
    momentum_u = np.zeros(u.shape)
    theta = np.random.uniform( 1.5, 4, size=(N))
    a = users.values[:,:]
    W_u = np.zeros((a.shape[1], K))
    momentum_W_u = np.zeros(W_u.shape)
    
    v = 0.1 * np.random.uniform(0,1, size=(M, K))
    momentum_v = np.zeros(v.shape)
    phi = np.random.uniform( 1.5, 4, size=(M))
    b = movies.values[:,:]
    W_v = np.zeros((b.shape[1], K))
    momentum_W_v = np.zeros(W_v.shape)
    
    #W_v = np.random.uniform( 0, 1, size=(b.shape[1], K))
    
    while itr_count > 0:
        random_row = randint(1,ratings.shape[0]) - 1;
        n = map_user_index[ratings[random_row,0]]
        m = map_movie_index[ratings[random_row,1]]
        obs_rating =  ratings[random_row,2]
        
        '''
            https://github.com/xuChenSJTU/PMF/blob/master/pmf_model.py
        
        '''
        # u[n] update equations
        grad_u_n = lambda_u * (u[n]- np.dot(a[n],W_u)) - lambda_x*(obs_rating - np.dot(u[n], v[m]) - theta[n] - phi[m])*v[m]
        momentum_u[n] = momentum * momentum_u[n] + learning_rate * grad_u_n
        u[n] =  u[n] - 0.01 * momentum_u[n]
        
        theta[n] = theta[n] + learning_rate*(lambda_x*(obs_rating - np.dot(u[n], v[m]) - theta[n] - phi[m]))
        
        grad_W_u =  -lambda_u*np.outer(a[n], (u[n] - np.dot(a[n],W_u)))
        momentum_W_u = momentum * momentum_W_u + 0.00001 * learning_rate * grad_W_u
        W_u =  W_u - momentum_W_u
        
        # v[n] update equations
        grad_v_m = lambda_v * (v[m] - np.dot(b[m],W_v)) - lambda_x*(obs_rating - np.dot(u[n], v[m]) - theta[n] - phi[m])*u[n]
        momentum_v[m] = momentum * momentum_v[m] + learning_rate * grad_v_m
        v[m] =  v[m] - 0.01 * momentum_v[m]
        
        phi[m] =  phi[m] + learning_rate*(lambda_x*(obs_rating - np.dot(u[n], v[m]) - theta[n] - phi[m]))
        
        grad_W_v = -lambda_v*np.outer(b[m],(v[m] - np.dot(b[m],W_v)))
        momentum_W_v = momentum * momentum_W_v + 0.00001 * learning_rate * grad_W_v
        W_v =  W_v - momentum_W_v
        
        itr_count = itr_count-1
        
    return u,theta,W_u,v,phi,W_v

def mf_alt_opt(ratings_raw, N, M, K, users, movies, lambda_u, lambda_v, lambda_x, itr_count):

    ratings = ratings_raw.values[:,0:3]
     
    # for finding correct index of a movie
    map_movie_index = get_movie_index_map(ratings_raw)
    map_user_index = get_user_index_map(ratings_raw)
    
    I_k = np.identity(K)
    userIDs = np.sort(np.asarray(ratings_raw['UserID'].unique()))
    movieIDs = np.sort(np.asarray(ratings_raw['MovieID'].unique()))
    
    # step 1 : Initialization u, theta , v, phi, W_u, W_v
    u = 0.1 * np.random.uniform(0,1, size=(N, K))
    theta = np.random.uniform( 1, 6, size=(N)) - 1
    a = users.values[:,:]
    W_u = np.zeros((a.shape[1], K))
    
    v = 0.1 * np.random.uniform(0,1, size=(M, K))
    phi = np.random.uniform(1, 6, size=(M)) - 1
    b = movies.values[:,:]
    W_v = np.zeros((b.shape[1], K))
    
    rmse = np.array([])
    while(itr_count > 0):
        
        # step 2 : u[n] and theta[n] for each n
        outer = np.zeros((K,K))
        t_error = np.zeros(K)
        for n in userIDs:
            r_n = ratings[ratings[:,0] == n]
            n = map_user_index[n]
            for i in r_n:
                m =  map_movie_index[i[1]]
                obs_rating = i[2]
                outer = outer + np.outer(v[m],v[m]) 
                t_error = t_error + (obs_rating - theta[n] - phi[m])*v[m]
            
            # u[n]
            u[n] = np.dot(np.linalg.inv(outer + I_k * (lambda_u/lambda_x)), ((lambda_u/lambda_x)*np.dot(a[n],W_u) + t_error))
            
            dev = 0
            for i in r_n:
                m =  map_movie_index[i[1]]
                obs_rating = i[2]
                dev =  dev + obs_rating - np.dot(u[n],v[m]) - phi[m]
                
            # theta[n]
            theta[n]  = dev/r_n.shape[0]
    
        # step 3 : v[m] and phi[m] for each m
        outer = np.zeros((K,K))
        t_error = np.zeros(K)
        for m in movieIDs:
            c_m = ratings[ratings[:,1] == m]
            m = map_movie_index[m]
            for i in c_m:
                n = map_user_index[i[0]]
                obs_rating = i[2]
                outer = outer + np.outer(u[n],u[n]) 
                t_error = t_error + (obs_rating - theta[n] - phi[m])*u[n]
            
            # v[m]
            v[m] = np.dot(np.linalg.inv(outer + I_k * (lambda_v/lambda_x)), ((lambda_v/lambda_x)*np.dot(b[m],W_v) + t_error))
            
            dev = 0
            for i in c_m:
                n = map_user_index[i[0]]
                obs_rating = i[2]
                dev =  dev + obs_rating - np.dot(u[n],v[m]) - theta[n]
                
            # phi[m]
            phi[m] = dev/c_m.shape[0]
            
        # step 4 : W_u and W_v
        W_u = np.linalg.multi_dot([np.linalg.inv(np.dot(a.T,a)),a.T,u])
        W_v = np.linalg.multi_dot([np.linalg.inv(np.dot(b.T,b)),b.T,v])
        
        # rmse for this iteration 
        rmse = np.append(rmse,evaluate_mf_sgd(ratings_raw, u, v, theta, phi))
        
        itr_count = itr_count -1
        
    return u,theta,W_u,v,phi,W_v,rmse
        
    
    
def mf_sgd(ratings_raw,N,M, K,contribute_per_step, learning_rate, lamba_u, lamba_v, itr_count ):
    
    ratings = ratings_raw.values[:,0:3]
    
    # for finding correct index of a movie
    map_movie_index = get_movie_index_map(ratings_raw)
    map_user_index = get_user_index_map(ratings_raw)
    
    # matrices
    u = np.random.uniform( 1, 5, size=(N, K))
    v = np.random.uniform(1, 5,size = (K, M)).T
    
    while itr_count > 0:
        random_row = randint(1,ratings.shape[0]) - 1
        n = map_user_index[ratings[random_row,0]]
        m = map_movie_index[ratings[random_row,1]]
        obs_rating =  ratings[random_row,2]
        
        u[n] = contribute_per_step*u[n] - learning_rate*(lamba_u * u[n] - (obs_rating - np.dot(u[n], v[m]))*v[m])
        v[m] = contribute_per_step*v[m] - learning_rate*(lamba_v * v[m] - (obs_rating - np.dot(u[n], v[m]))*u[n])
        
        itr_count = itr_count-1
    
    return u, v

def evaluate_mf_sgd(ratings_raw, u, v, theta , phi):
     
    # for finding correct index of a movie
    map_movie_index = get_movie_index_map(ratings_raw)
    map_user_index = get_user_index_map(ratings_raw)
    
    ratings = ratings_raw.values[:,0:3]
    sum_of_sq_error = 0.0
    for row in ratings:
        n = map_user_index[row[0]]
        m = map_movie_index[row[1]]
        obs_rating = row[2]
        squared_error = (obs_rating - np.dot(u[n],v[m]) - theta[n] - phi[m])**2
        sum_of_sq_error = sum_of_sq_error + squared_error
    RMSE = np.sqrt(sum_of_sq_error/ratings.shape[0])
    return RMSE


def ridge_regression(X, y, alpha):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0) 
    
    # Ridge Regression
    from sklearn.linear_model import Ridge
    regressor = Ridge(alpha)
    regressor.fit(X_train,y_train)
    
    y_pred_train = regressor.predict(X_train)
    y_pred_test = regressor.predict(X_test)
    
    # rmse
    rmse_train = np.sqrt(np.mean((y_train - y_pred_train)**2))
    rmse_test = np.sqrt(np.mean((y_test - y_pred_test)**2))
    
    return regressor, rmse_train, rmse_test

def visualize_pca_features(X_raw, K):
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_raw)
    
    # PCA Reduction
    from sklearn.decomposition import PCA # linear dimentionality reduction
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_sc)
    
    # visualize
    plt.scatter(X_pca[:,0],X_pca[:,1])
    plt.xlabel('First PCA Component')
    plt.ylabel('Second PCA Component')
    plt.title('When Number of Latent Features = ' + str(K))
    plt.savefig('pca_embedding_for_users_k_'+str(K)+'.png')
    plt.show()
    
    return X_pca

def visualize_tsne_features(X_raw, K):    
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_raw)
    
    # tsne reduction
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X_sc)
    
     # visualize
    plt.scatter(X_tsne[:,0],X_tsne[:,1])
    plt.xlabel('First t-SNE Component')
    plt.ylabel('Second t-SNE Component')
    plt.title('When Number of Latent Features = ' + str(K))
    plt.savefig('tsne_embedding_for_users_k_'+str(K)+'.png')
    plt.show()
    
    return X_tsne

def show_dendogram(X, K):
    
    # figuring optimal number of cluster via dendograms
    import scipy.cluster.hierarchy as sch
    dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

    # ward to minimize variance (similar to WCSS in KMeans)
    
    plt.title('Dendogram Case : When K = ' + str(K))
    plt.xlabel('Clusters Number')
    plt.ylabel('Euclidean Distance')
    plt.savefig('dendogram_on_users_latent_feat_pca_K_'+str(K)+'.png')
    plt.show()

def get_clusters(X,num_clusters):
    
    # hierarchical clustering
    
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = num_clusters, affinity = 'euclidean',
                         linkage = 'ward')
    
    clusters = hc.fit_predict(X)
    '''
    # K means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = num_clusters, init = 'k-means++' , n_init = 10 , max_iter = 300,
                random_state = 0)
    clusters = kmeans.fit_predict(X)
    '''
    return clusters

#def predict_cluster():
    
    
def visualize_clusters(X, clusters, num_clusters, K):
    for C in range(0,num_clusters):
        plt.scatter(X[clusters==C,0],X[clusters==C,1])
        
    plt.title('Hierarchical Clustering on users with pca latent features, \nwhen K = '+str(K)+' and #clusters = '+str(num_clusters))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.savefig('HC_on_users_tsne_K_'+str(K)+'.png')
    plt.show()
    
def get_classifier(X,y):
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10,
                                                    random_state = 0)
    
    # feature scaling 
    from sklearn.preprocessing import StandardScaler
    scale_X = StandardScaler()
    X_train = scale_X.fit_transform(X_train)
    X_test = scale_X.transform(X_test)
    
    # Fitting classifier to the Training set
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', 
                                    random_state = 0)
    classifier.fit(X_train,y_train)
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return classifier, cm

        
def get_movie_index_map(ratings_raw):
    lst_movieIDs = np.sort(np.asarray(ratings_raw['MovieID'].unique()))
    lst = []
    for i in range(0,len(lst_movieIDs)):
        lst.append((lst_movieIDs[i],i)) 
    return dict(lst)


def get_user_index_map(ratings_raw):
    lst_UserIDs = np.sort(np.asarray(ratings_raw['UserID'].unique()))
    lst = []
    for i in range(0,len(lst_UserIDs)):
        lst.append((lst_UserIDs[i],i)) 
    return dict(lst)

