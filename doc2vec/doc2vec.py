import pickle
from time import time

import gensim
from keras import Input, models, optimizers
from scrapy.spiders import CrawlSpider

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
from tqdm import tqdm
import numpy as np


X_train= pickle.load(open("../data_final/X_train.pkl",'rb'))
y_train= pickle.load(open("../data_final/y_train.pkl",'rb'))

X_test= pickle.load(open("../data_final/X_test.pkl",'rb'))
y_test= pickle.load(open("../data_final/y_test.pkl",'rb'))


def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3):
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)

    if is_neuralnet:
        classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=512)

        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)
        val_predictions = val_predictions.argmax(axis=-1)
        test_predictions = test_predictions.argmax(axis=-1)
    else:
        classifier.fit(X_train, y_train)

        train_predictions = classifier.predict(X_train)
        val_predictions = classifier.predict(X_val)
        test_predictions = classifier.predict(X_test)

    print("Validation accuracy: ", metrics.accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", metrics.accuracy_score(test_predictions, y_test))


def create_dnn_model():
    input_layer = Input(shape=(300,))
    layer = Dense(1024, activation='relu')(input_layer)
    layer = Dense(1024, activation='relu')(layer)
    layer = Dense(512, activation='relu')(layer)
    output_layer = Dense(3, activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return classifier

def get_corpus(documents):
    corpus = []

    for i in tqdm(range(len(documents))):
        doc = documents[i]

        words = doc.split(' ')
        tagged_document = gensim.models.doc2vec.TaggedDocument(words, [i])

        corpus.append(tagged_document)

    return corpus


train_corpus = get_corpus(X_train)
test_corpus = get_corpus(X_test)


model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
model.build_vocab(train_corpus)

# %time model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)


X_data_vectors = []
for x in train_corpus:
    vector = model.infer_vector(x.words)
    X_data_vectors.append(vector)

X_test_vectors = []
for x in test_corpus:
    vector = model.infer_vector(x.words)
    X_test_vectors.append(vector)


#
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
print("running....")
classifier = create_dnn_model()
train_model(classifier=classifier, X_data=np.array(X_data_vectors), y_data=y_train, X_test=np.array(X_test_vectors), y_test=y_test, is_neuralnet=True, n_epochs=5)

# classifier.fit(np.array(X_data_vectors),y_train,epochs=5)
# y_predict = model.predict(X_test)
# # print(y_predict)
# y_predict = y_predict.argmax(axis=-1)
# y_test = y_test.argmax(axis=-1)
# print(y_test)
# print("Test accuracy: ", metrics.accuracy_score(y_predict, y_test))
# print("Report:",classification_report(y_test,y_predict))


