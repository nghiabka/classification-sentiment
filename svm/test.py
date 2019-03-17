import pickle
import re

import gensim
from pyvi import ViTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



def get_dict_word( path):
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
def sua_chinh_ta( str):
    str = str.split(" ")
    list_word = get_dict_word("../chinhta")
    for i in range(len(str)):
        if str[i] in list_word:
            str[i] = list_word.get(str[i])
    return " ".join(str)


# ghep tu
def gheptu( sentent):
    sentent = sentent.split(" ")
    # print(sentent)
    list_phu_dinh = ['không', 'chẳng', 'chả', 'đâu', 'đâu_có', 'khỏi', 'chưa', 'đếch', 'đéo']
    for i in range(len(sentent)):
        if sentent[i] in list_phu_dinh and i + 1 < len(sentent):
            sentent[i + 1] = sentent[i] + "_" + sentent[i + 1]
            sentent[i] = ''
    for i in range(len(sentent)):
        if sentent[i] in list_phu_dinh and i + 1 < len(sentent):
            sentent[i + 1] = sentent[i] + sentent[i + 1]
            sentent[i] = ''

    return " ".join(sentent)


# delete email , number , links,specia char


def remove_link_and_email(sentence):
    sentence = sentence.lower()
    # sentence = self.gheptu(sentence)

    # remove link and email
    sentence = re.sub(r"(\S+@\S+)", "", sentence)
    sentence = re.sub(r"www\S+", "", sentence)
    sentence = re.sub(r'(http\S+)', "", sentence)
    return sentence


# loai bo stop word
def remove_stop_word(sentence, stopwords):
    sentence = sentence.split(" ")
    sentence = [word for word in sentence if word not in stopwords]

    return " ".join(sentence)


# tach tu va loai bo cac ki tu dac biet
def token_and_remove_special_char(sentence):
    sentence = ViTokenizer.tokenize(sentence)
    # loại bỏ các kí tự đặc biệt
    sentence = gensim.utils.simple_preprocess(sentence)
    sentence = " ".join(sentence)
    return sentence


def process_text(sentence):
    stopwords = open("../stopwords", "r").read().split("\n")
    # sua chinh ta 1 so tu sai pho bien

    sentence = sua_chinh_ta(sentence)
    # sentence = gheptu(sentence)
    # remove link and email
    sentence = remove_link_and_email(sentence)
    # tach từ và loại bỏ các kí tự đăc biệt
    sentence = token_and_remove_special_char(sentence)
    # remove stop word
    sentence = remove_stop_word(sentence, stopwords)
    return sentence


if __name__ == '__main__':
    model = pickle.load(open("./svm_model.pkl", "rb"))
    couter = pickle.load(open("./counter.pkl","rb"))
    tfidf = pickle.load(open("./tfidf.pkl","rb"))

    # str ="tôi không thích nó"
    # str = "con chó này đáng ghét"
    str  = "bphone như cứt"
    str = "tôi chưa đủ tiền mua xe hơi"
    str = "vinamil làm ăn như cứt"
    str = "viettel làm ăn chán bỏ mẹ"
    str = "đội bạn đá như cục cứt"
    str = "hôm nay trời đẹp "
    str = "bphone làm ăn như cục cứt"
    str= "3###,ad ơi, cho mình hỏi, mình có tk smast của vp bank, cái online trên mạng ấy ạ, mình muốn nạp tiền vào thì nạp bằng cách nào ạ? có mất phí cụ thể như thế nào không, ad trả lời giúp mình với "
    str ="con chó không đẹp"



    X_data =process_text(str)
    X_data = couter.transform([X_data])
    X_data = tfidf.transform(X_data)
    result = model.predict(X_data)
    if result[0] == 0:
        print("positive")
    elif result[0] == 1:
        print("negative")
    else:
        print("neutural")
    
    while 1:
         print("hãy nhập 1 câu có dấu: ")
         str = input()
         if str == "exit":
             break

    
         X_data =process_text(str)
         X_data = couter.transform([X_data])
         X_data = tfidf.transform(X_data)
         result = model.predict(X_data)
         if result[0] ==0:
             print("POSITIVE")
         elif result[0] ==1:
             print("NEGATIVE")
         else:
             print("NEUTURAL")
    

