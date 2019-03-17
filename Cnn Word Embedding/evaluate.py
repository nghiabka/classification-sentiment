

import pickle
from itertools import cycle

from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize

model = pickle.load(open("./cnn2.pkl",'rb'))



X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))

print(y_test)

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)


def process_doc(text, vocab):
    documents= list()
    for doc in text:
        doc = doc.split(" ")
        tokens = [w for w in doc if w in vocab]
        tokens = ' '.join(tokens)
        documents.append(tokens)
    return documents
X_train = process_doc(X_train,vocab)
X_test  = process_doc(X_test,vocab)
print(y_test)


# create the tokenizer
tokenizer = Tokenizer()
# fit the tokenizer on the documents
tokenizer.fit_on_texts(X_train)
# sequence encode
encoded_docs = tokenizer.texts_to_sequences(X_train)
print(X_train)
# print(X_train)
# print(Counter(y_train))

# pad sequences
max_length = max([len(s.split()) for s in X_train])
print(max_length)
Xtrain = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(Xtrain)
encoded_docs = tokenizer.texts_to_sequences(X_test)
# pad sequences
Xtest = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(len(Xtrain[0]))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
vocab_size = len(tokenizer.word_index) + 1
print(max_length)
loss,acc = model.evaluate(Xtest,y_test)
print(acc)
y_predict = model.predict(Xtest)
print(y_predict)
y_predict = y_predict.argmax(axis=-1)
y_test = y_test.argmax(axis=-1)



print("Validation accuracy: ", metrics.accuracy_score(y_predict, y_test))
print("Test accuracy: ", metrics.accuracy_score(y_predict, y_test))
print("Report:",classification_report(y_test,y_predict))



from sklearn.metrics import roc_curve
y_pred_keras = model.predict(Xtest)
y_test = label_binarize(y_test, classes=[0, 1, 2])
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_keras[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])



n_classes =3

lw =2

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred_keras.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('classification sentiment')
plt.legend(loc="lower right")
plt.show()


# # Zoom in view of the upper left corner.
# plt.figure(2)
# plt.xlim(0, 0.2)
# plt.ylim(0.8, 1)
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)
#
# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)
#
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(i, roc_auc[i]))
#
# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Some extension of Receiver operating characteristic to multi-class')
# plt.legend(loc="lower right")
# plt.show()