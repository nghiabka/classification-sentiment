print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
import pickle
from sklearn.feature_extraction.text import CountVectorizer# Import some data to play with
from sklearn.feature_extraction.text import TfidfTransformer

# print(y)
# load data
X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))


#

#
y_train =label_binarize(y_train, classes=[0,1,2])
y_test = label_binarize(y_test,classes=[0,1,2])
'''
    {'tfidf__norm': 'l2', 'vect__max_df': 0.5, 'clf__C': 2, 'vect__max_features': None, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}

'''


# Count Vectors as features
counter = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=None, max_df=0.5)
counter.fit(X_train)
X_train = counter.transform(X_train)
X_test = counter.transform(X_test)

# use tfidf
tfidf = TfidfTransformer(use_idf=True, norm='l2')
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)



# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm.SVC(kernel='rbf', probability=True,C =100,gamma=0.01))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
y_pred =classifier.predict(X_test)

print('F1 Score:', np.round(metrics.f1_score(y_test, y_pred, average='micro'), 3))

# roc, area
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC pos (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='red',
         lw=lw, label='ROC neg (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='green',
         lw=lw, label='ROC neu (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()