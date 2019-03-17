from collections import Counter
from time import time
from sklearn.metrics import roc_curve, auc,confusion_matrix
import pandas
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.utils.multiclass import type_of_target
from sklearn.preprocessing import label_binarize





#  text feature extraction parameters
def extraction_param_selection(X, y, nfolds,model):
    print("running....")
    pipeline = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf',model)])
    # Cs = [0.001, 0.01, 0.1, 1, 10]
    #
    # gammas = [0.001, 0.01, 0.1, 1,2]
    parameters = {
        'vect__analyzer':('word','char','char_wb'),
        'vect__max_df': (0.5, 0.75, 1.0),
        'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__analyzer': ('word', 'char', 'char_wb'),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        # 'clf__gamma':gammas,
        # 'clf__C': Cs
    }

    # param_grid = {'C': Cs}
    grid_search = GridSearchCV(pipeline, parameters, cv=nfolds,n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search




def grid_search_param_svm(kernel, X,y, kfolds):
    print("running....")
    pipeline = Pipeline([('clf', svm.SVC())])
    Cs = [0.001, 0.01, 0.1, 1, 10, 100]

    gammas = [0.001, 0.01, 0.1, 1,10,]
    coef0s = [0, 1,2]
    degrees =[2,5]
    if kernel =="rbf":
        parameters = {
            'clf__kernel': ['rbf'],
            'clf__C': Cs,
            'clf__gamma': gammas,
        }
        grid_search = GridSearchCV(pipeline, parameters, cv=kfolds, n_jobs=-1)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        return grid_search
    elif kernel =="linear":
        parameters = {
            'clf__kernel': ['linear'],
            'clf__C': Cs,
            'clf__gamma':gammas
        }
        grid_search = GridSearchCV(pipeline, parameters, cv=kfolds, n_jobs=-1)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        return grid_search
    elif kernel =="poly":
        parameters = {
            'clf__kernel': ['poly'],
            'clf__C': Cs,
            'clf__gamma': gammas,
            'clf__coef0':coef0s,
            'clf__degree':degrees

        }
        grid_search = GridSearchCV(pipeline, parameters, cv=kfolds, n_jobs=-1)
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        return grid_search





if __name__ == '__main__':
    '''
        {'tfidf__norm': 'l2', 'vect__max_df': 0.5, 'clf__C': 2, 'vect__max_features': None, 'tfidf__use_idf': True, 
        'vect__ngram_range': (1, 2),vect__analyzer='word'}

    '''
    # load data
    X_train = pickle.load(open("../data_final/X_train.pkl", 'rb'))
    y_train = pickle.load(open("../data_final/y_train.pkl", 'rb'))

    X_test = pickle.load(open("../data_final/X_test.pkl", 'rb'))
    y_test = pickle.load(open("../data_final/y_test.pkl", 'rb'))

    counter = CountVectorizer(analyzer='word',ngram_range=(1, 2), max_features=None, max_df=0.5)
    counter.fit(X_train)
    X_train = counter.transform(X_train)
    X_test = counter.transform(X_test)
    tfidf = TfidfTransformer(use_idf=True, norm='l2')
    tfidf.fit(X_train)
    X_train = tfidf.transform(X_train)
    X_test = tfidf.transform(X_test)




    t0 = time()
    # grid_search = grid_search_param_svm('rbf',X_train,y_train,5)
    grid_search = grid_search_param_svm('linear',X_train,y_train,5)
    # grid_search = extraction_param_selection(X,y_data,5,svm.SVC(kernel='rbf',C =100, gamma=0.01))
    print("done in %0.3fs" % (time() - t0))

    print("Best score: %0.3f" % grid_search.best_score_)
    print("score for test : %0.3f"  % grid_search.score(X_test,y_test))



    # print("Best parameters set:")
    # best_parameters = grid_search.best_estimator_.get_params()
    # print(best_parameters)

    '''
   {'{'clf__kernel': 'rbf', 'clf__gamma': 0.01, 'clf__C': 100}
done in 801.588s
Best score: 0.748
score for test : 0.771
    '''
    '''
{'clf__gamma': 0.001, 'clf__kernel': 'linear', 'clf__C': 100}
done in 275.712s
Best score: 0.739
score for test : 0.761


    '''


