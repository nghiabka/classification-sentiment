import pickle
import numpy as np
from sklearn import svm, metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import pickle
from sklearn.model_selection import cross_val_score
from collections import Counter



# load data
X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))



'''
    {'tfidf__norm': 'l2', 'vect__max_df': 0.5, 'clf__C': 2, 'vect__max_features': None, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}

'''

# Count Vectors as features
counter = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=None, max_df=0.5)
counter.fit(X_train)
X_train = counter.transform(X_train)
X_test = counter.transform(X_test)


tfidf = TfidfTransformer(use_idf=True, norm='l2')
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)




# train model
# model = SGDClassifier(penalty = 'l2', alpha = 0.001, n_jobs =  -1, loss = 'hinge')

model = svm.SVC(kernel='rbf',C =100,gamma=0.01)

model.fit(X_train,y_train)

scores = cross_val_score(model, X_test, y_test, cv=5)
print("cross_vali_score:",scores)

# Result
y_predict =model.predict(X_test)

print("Validation accuracy: ", metrics.accuracy_score(y_predict, y_test))
print("Report:",classification_report(y_test,y_predict))

# save model.couter , tfidf
pickle.dump(model,open("./svm_model.pkl","wb"))
pickle.dump(counter,open("counter.pkl","wb"))
pickle.dump(tfidf,open("tfidf.pkl","wb"))



'''
cross_vali_score: [ 0.72404372  0.72876712  0.74450549  0.75549451  0.73076923]
Validation accuracy:  0.773450356555
Report:              precision    recall  f1-score   support

          0       0.77      0.77      0.77       592
          1       0.80      0.81      0.81       610
          2       0.75      0.74      0.74       621

avg / total       0.77      0.77      0.77      1823
    '''
