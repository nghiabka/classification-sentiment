import pickle

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical


def creat_dnn():
    model = Sequential()
    model.add(Dense(2048, input_shape=(61886,), activation='relu'))
    model.add(Dense(1024,  activation='relu'))

    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return  model


X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))
def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

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


counter = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=None, max_df=0.5)
counter.fit(X_train)
X_train = counter.transform(X_train)
X_test = counter.transform(X_test)


tfidf = TfidfTransformer(use_idf=True, norm='l2')
tfidf.fit(X_train)
X_train = tfidf.transform(X_train)
X_test = tfidf.transform(X_test)
y_train =label_binarize(y_train, classes=[0,1,2])
y_test = label_binarize(y_test,classes=[0,1,2])
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
model = creat_dnn()
print("running....")
model.fit(X_train,y_train)

# save model
pickle.dump(model,open("./dnn_model.pkl","wb"))


y_predict = model.predict(X_test)
# print(y_predict)
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(acc)
y_predict = y_predict.argmax(axis=-1)
y_test = y_test.argmax(axis=-1)
print(y_test)
# print(acc)


# y_predict = y_predict.argmax(axis=-1)
# print("Validation accuracy: ", metrics.accuracy_score(y_predict, y_test))
# print(model.score(X_test,y_test))
print("Test accuracy: ", metrics.accuracy_score(y_predict, y_test))
print("Report:",classification_report(y_test,y_predict))
'''
Test accuracy:  0.773450356555
Report:              precision    recall  f1-score   support

          0       0.81      0.73      0.77       592
          1       0.81      0.81      0.81       610
          2       0.71      0.78      0.74       621

avg / total       0.78      0.77      0.77      1823
'''