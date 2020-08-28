import pickle
import re
from random import shuffle
from collections import Counter
# import gensim
import gensim
from pyvi.ViTokenizer import ViTokenizer
import os
import csv

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas

class PreprocessData():

    def __init__(self,path):
        self.path = path
    # load cac tu dien cac tu sai chinh ta
    def get_dict_word(self,path):
        fr = open(path, "r")
        list_word = dict()
        for line in fr:
            # print(line)
            line = line.split(":")
            value = line[0]
            keys = line[1].rstrip("\n").split(",")
            for key in keys:
                k_v = {key: value}
                list_word.update(k_v)

        return list_word
    # sua 1 tu sai chinh ta
    def sua_chinh_ta(self,str):
        str = str.split(" ")
        list_word = self.get_dict_word("chinhta")
        for i in range(len(str)):
            if str[i] in list_word:
                str[i] = list_word.get(str[i])
        return " ".join(str)

    # ghep tu
    def gheptu(self,sentent):
        sentent = sentent.split(" ")
        # print(sentent)
        list_phu_dinh = ['không','chẳng','chả','đâu','đâu_có','khỏi','chưa','đếch','đéo']
        for i in range(len(sentent)):
            if sentent[i] in list_phu_dinh and i+1<len(sentent):
                sentent[i + 1] = sentent[i] +"_"+ sentent[i + 1]
                sentent[i] = ''
        for i in range(len(sentent)):
            if sentent[i] in list_phu_dinh and i+1<len(sentent):
                sentent[i + 1] = sentent[i] + sentent[i + 1]
                sentent[i] = ''

        return " ".join(sentent)


    # delete email , number , links,specia char


    def remove_link_and_email(self,sentence):
        sentence = sentence.lower()
        sentence = self.sua_chinh_ta(sentence)
        # sentence = self.gheptu(sentence)

        # remove link and email
        sentence = re.sub(r"(\S+@\S+)","",sentence)
        sentence = re.sub(r"www\S+","",sentence)
        sentence = re.sub(r'(http\S+)',"",sentence)
        return sentence



    # loai bo stop word
    def remove_stop_word(self, sentence, stopwords):
        sentence = sentence.split(" ")
        sentence = [word for word in sentence if word not in stopwords]

        return " ".join(sentence)

    # tach tu va loai bo cac ki tu dac biet
    def token_and_remove_special_char(self,sentence):
        sentence = ViTokenizer.tokenize(sentence)
        # loại bỏ các kí tự đặc biệt
        sentence = gensim.utils.simple_preprocess(sentence)
        sentence = " ".join(sentence)
        return sentence
    # xu lý câu
    def process_text(self,sentence):
        stopwords = open("stopwords","r").read().split("\n")
        # sua chinh ta 1 so tu sai pho bien

        sentence = self.sua_chinh_ta(sentence)
        # sentence = self.gheptu(sentence)
        # remove link and email
        sentence = self.remove_link_and_email(sentence)
        # tach từ và loại bỏ các kí tự đăc biệt
        sentence = self.token_and_remove_special_char(sentence)
        # remove stop word
        sentence = self.remove_stop_word(sentence,stopwords)
        return sentence




    def proceess(self):

        # load du lieu goc chua xu ly
        try:

            fr = open("./data_final/data_origin.csv")
        except:
            print("canot open file origin data")
            return -1

        # luu du lieu da xu ly
        csvfile = open('./data_final/data.csv', 'w')
        fieldnames = ['label', 'data']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # xu ly
        for line in tqdm(fr):
            # print(line)
            if line[0] =='/':
                continue
            line = line.split(",",1)
            print(line)
            if line[0] != '1' and line[0] != '2' and line[0] != '3':
                continue
            label =line[0]
            print(label)
            #  tien xu ly cau
            data = self.process_text(line[1])
            writer.writerow({'label':label,'data':data})

        fr.close()
        csvfile.close()




if __name__ == '__main__':

    pre = PreprocessData("./data_final/data_origin.csv")

    print(pre.proceess())


    data = pandas.read_csv('./data_final/data.csv').values
    print(data)


    y = data[:, 0]
    print(Counter(y))
    X = data[:, 1]
    positive = []
    negative = []
    neutural = []
    for i in range(len(X)):
        if y[i] == 1:
            positive.append(X[i])
        elif y[i] == 2:
            negative.append(X[i])
        elif y[i] == 3:
            neutural.append(X[i])

    positive =positive[:1841]
    negative = negative[:1841]
    neutural = neutural[:1841]
    # label
    y_data = []
    X_data = []
    for doc in positive:
        X_data.append(doc)
        y_data.append(0)
    for doc in negative:
        X_data.append(doc)
        y_data.append(1)
    for doc in neutural:
        X_data.append(doc)
        y_data.append(2)

    # X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.33)
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.33)


    print(Counter(y_train))
    print(Counter(y_test))

    # luu du lieu
    pickle.dump(X_train, open("./data_final/X_train.pkl", "wb"))
    pickle.dump(y_train, open("./data_final/y_train.pkl", "wb"))
    pickle.dump(X_test, open("./data_final/X_test.pkl", "wb"))
    pickle.dump(y_test, open("./data_final/y_test.pkl", "wb"))
#no come	
'''
'''


