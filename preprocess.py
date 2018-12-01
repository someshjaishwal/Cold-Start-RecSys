import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def user_info(users_raw,ratings_raw = False):
    users_raw = users_raw.merge(pd.DataFrame(ratings_raw['UserID'].unique(),columns=['UserID']) , how = 'right', left_on = 'UserID', right_on = 'UserID')
    return process_user_info(users_raw)

def process_user_info(df):
    
    feature_list_new=['UserID','Gender','Age','Occupation_0','Occupation_1','Occupation_2','Occupation_3','Occupation_4','Occupation_5','Occupation_6','Occupation_7','Occupation_8','Occupation_9','Occupation_10','Occupation_11','Occupation_12','Occupation_13','Occupation_14','Occupation_15','Occupation_16','Occupation_17','Occupation_18','Occupation_19','Occupation_20','Zip-code']
    zero_data = np.zeros(shape=(len(df),len(feature_list_new)))
    df1= pd.DataFrame(zero_data, columns=feature_list_new)
    
    for i in range(len(df)):
            if df.at[i,'Gender']=='M':
                    df1.at[i,'Gender']=1
            elif df.at[i,'Gender']=='F':
                    df1.at[i,'Gender']=0
            df1.at[i,'UserID'] = df.at[i,'UserID']
            df1.at[i,'Age'] =  df.at[i,'Age']
            df1.at[i,'Zip-code']=df.at[i,'Zip-code'].split('-')[0]
            occupation = "Occupation_" + str(df.at[i,'Occupation'])
            df1.at[i,occupation] = 1
    df1 = df1.drop(['Occupation_0'],axis=1)
    
    return df1


def new_user_info(user):
    df = pd.DataFrame([user],columns=['UserID','Gender','Age','Occupation','Zip-code'])  
    return process_user_info(df)

def movie_info(movies_raw, ratings_raw):
    
    # filter out details of rated movie    
    movies_raw = movies_raw.merge(pd.DataFrame(ratings_raw['MovieID'].unique(),columns=['MovieID']) , how = 'right', left_on = 'MovieID', right_on = 'MovieID')
    movies = movies_raw.drop('Title', axis=1)
    movies['Genres'] = movies['Genres'].str.split('|')
    movies = movies['Genres'].apply(pd.Series).merge(movies, left_index = True, right_index = True)
    movies = movies.drop(["Genres"], axis = 1).melt(id_vars ='MovieID', value_name = 'Genres')
    movies = movies.drop("variable", axis = 1).dropna()
    
    # store genres
    genres = np.array(np.sort(np.asarray(movies['Genres'].unique())))
    
    # encoding catgorial data
    X = movies.values
    lblEncX = LabelEncoder()
    X[:,1] = lblEncX.fit_transform(X[:,1])
    X = pd.DataFrame(X)
    ones =pd.Series(np.ones(len(X.index)))
    X['Ones'] = ones
    X.columns = ['MovieID', 'Genres', 'Ones']
    X = X.pivot(index='MovieID', columns='Genres', values='Ones')
    X.fillna(0, inplace=True)
    movies = X.reset_index(drop=True)
    
    return movies, genres

def new_movie_info_2(movie):
    movie = movie.drop('Title', axis=1)
    movie['Genres'] = movie['Genres'].str.split('|')
    movie = movie['Genres'].apply(pd.Series).merge(movie, left_index = True, right_index = True)
    movie = movie.drop(["Genres"], axis = 1).melt(id_vars ='MovieID', value_name = 'Genres')
    movie = movie.drop("variable", axis = 1).dropna()
    
    X = movie.values
    lblEncX = LabelEncoder()
    X[:,1] = lblEncX.fit_transform(X[:,1])
    X = pd.DataFrame(X)
    ones =pd.Series(np.ones(len(X.index)))
    X['Ones'] = ones
    X.columns = ['MovieID', 'Genres', 'Ones']
    X = X.pivot(index='MovieID', columns='Genres', values='Ones')
    X.fillna(0, inplace=True)
    movie = X.reset_index(drop=True)
    
    return movie
    
def new_movie_info(movie, genre_mapping):
    movie_genres = movie[2].split('|')
    movie = np.zeros(len(genre_mapping))
    for genre in movie_genres:
        movie[genre_mapping[genre]] = 1
    
    return movie
    
 
def split(ratings_raw, fraction = 0.1):
    ratings_nan = ratings_raw.iloc[:,0:4]
    ratings_nan = ratings_nan.pivot(index='UserID', columns='MovieID', values='Rating')
    
    movies_test = ratings_nan.sample(frac=0.1, axis = 1)
    ratings_nan = ratings_nan.drop(movies_test.columns.values, axis = 1)
 
    users_test = ratings_nan.sample(frac=0.1, axis = 0)
    ratings_nan = ratings_nan.drop(users_test.index, axis = 0)
    movies_test = movies_test.drop(users_test.index, axis = 0)

    ratings_nan = ratings_nan.reset_index()
    ratings_train_raw = pd.melt(ratings_nan, id_vars=["UserID"], var_name="MovieID" , value_name = "Rating").dropna()
    ratings_train_raw['MovieID'] = ratings_train_raw['MovieID'].astype('int64')
    
    movies_test = movies_test.reset_index()
    movies_test = pd.melt(movies_test, id_vars=["UserID"], var_name="MovieID", value_name = "Rating").dropna()

    users_test = users_test.reset_index()
    users_test = pd.melt(users_test, id_vars=["UserID"], var_name="MovieID", value_name = "Rating").dropna()
    
    return ratings_train_raw, users_test , movies_test

