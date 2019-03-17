import pickle
from collections import Counter

import keras
from keras import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Conv2D
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn import metrics, preprocessing
from sklearn.metrics import classification_report,f1_score

from twisted.conch.insults.insults import privateModes

X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))
a = y_test


vocab = Counter()



def update_vocab(text,vocab):
    for doc in text:
        doc = doc.split(" ")
        vocab.update(doc)



update_vocab(X_train,vocab)
update_vocab(X_test,vocab)
print(len(vocab))
print(vocab)
min_occurane = 1
tokens = [k for k,c in vocab.items() if c >= min_occurane]
# print(len(tokens))


# save list to file
def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w')
    # write text
    file.write(data)
    # close file
    file.close()


save_list(tokens, 'vocab.txt')


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
# encoder = preprocessing.LabelEncoder()
# y_train = encoder.fit(y_train)

vocab_size = len(tokenizer.word_index) + 1
print(max_length)
# print(vocab_size)
#
# define model
model = Sequential()
model.add(Embedding(vocab_size, 2000, input_length=max_length))

model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(MaxPooling1D(pool_size=5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))
print(model.summary())

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(Xtrain, y_train, epochs=10, verbose=2,batch_size=100)
# evaluate
loss, acc = model.evaluate(Xtest, y_test, verbose=0)



print(acc)

pickle.dump(model,open("./cnn2.pkl","wb"))

y_predict = model.predict(Xtest,verbose=2)
#
# for num in y_predict:
#     print(round(num[0],1))


y_predict = y_predict.argmax(axis=-1)
y_test = y_test.argmax(axis=-1)
# print(y_predict)
print('Test Accuracy: %f' % (acc*100))
print("Test accuracy: ", metrics.accuracy_score(y_predict, y_test))
print("Report:",classification_report(y_test,y_predict))
print()
# # print(loss)
print("f1 sccore:",f1_score(y_predict,y_test,average=None))
'''
Test accuracy:  0.765770707625
Report:              precision    recall  f1-score   support

          0       0.77      0.77      0.77       592
          1       0.80      0.78      0.79       610
          2       0.74      0.75      0.74       621

avg / total       0.77      0.77      0.77      1823


f1 sccore: [ 0.76728499  0.78943022  0.74139311]
'''