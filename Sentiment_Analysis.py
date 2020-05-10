from gensim.models import Word2Vec
import pandas as pd
import re as re
import numpy as np
import random as random
from collections import Counter

size_vec = 100

def read_data(file_name = 'tweets.csv'):
    data = pd.read_csv(file_name)
    texts = data['text'].to_list()
    sentiment = data['airline_sentiment'].map({'positive': 1, 'negative': 0 , 'neutral': 2}).to_list()
    return texts , sentiment

def re_sub_data(data_text):
    sentences = []
    for text in data_text:
        tokens = re.sub(r"[^a-z0-9]+"," " , text.lower()).split()
        sentences.append(tokens)
    return sentences

def build_Word2Vec(sentences):
    model = Word2Vec(sentences , size= size_vec , window = 5 , min_count= 4 , workers= 4 , sg = 0 )
    return model   

def get_vocab_list(model):
    vocab_list = np.array(list(model.wv.vocab.keys()))
    return vocab_list

def get_average_sentence(sentence , model):
    vocabs = get_vocab_list(model)
    words = np.intersect1d(sentence , vocabs)

    if words.shape[0] > 0:
        words_add = np.sum(model.wv[words] , axis=0)
        average = np.divide(words_add , float(len(sentence)))
        return average #np.mean(model.wv[sentence], axis=0).tolist()
    else:
        return np.zeros(size_vec).tolist()
    
def get_average_sentences(all_sentences , model , sentiment):
    all_avg = list()
    index = 0 
    for sen in all_sentences:
        sen_avg = get_average_sentence(sen , model)
        all_avg.append( (sen_avg , sentiment[index]) )
        index += 1
    return all_avg

def get_sum_sentence(sentence , model):
    vocabs = get_vocab_list(model)
    words = np.intersect1d(sentence , vocabs)

    if words.shape[0] > 0:
        words_add = np.sum(model.wv[words] , axis=0)
        return words_add #np.mean(model.wv[sentence], axis=0).tolist()
    else:
        return np.zeros(size_vec).tolist()
    
def get_sum_sentences(all_sentences , model , sentiment):
    all_sum = list()
    index = 0 
    for sen in all_sentences:
        sen_sum = get_sum_sentence(sen , model)
        all_sum.append( (sen_sum , sentiment[index]) )
        index += 1
    return all_sum
   
def divide_data(data):
    random.shuffle(data)    
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(train_data)):int(len(data)*1)]
    return train_data , test_data

def split_data(X_Y_data):
    X = [row[0] for row in X_Y_data]
    Y = [row[1] for row in X_Y_data]
    return X , Y

def init_data(data):
    train_data , test_data = divide_data(data)
    train_X , train_Y = split_data(train_data)
    test_X , test_Y = split_data(test_data)
    return train_X , train_Y , test_X , test_Y


texts , sentiment = read_data()

'''
def sort_by_sentiment(sentences):
    sentences.sort(key = lambda x:x[1])
    return sentences


text , sentiment = read_data()
sentences = re_sub_data(text)
model = build_Word2Vec(sentences)
average = get_average_sentences( sentences , model , sentiment )
sentences = sort_by_sentiment(average)
d0 = list(item[1] for item in sentences).count(0)
d1 = list(item[1] for item in sentences).count(1)
d2 = list(item[1] for item in sentences).count(2)
print(d0)
print(d1)
print(d2)
print(d0+d1+d2)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
sentiment = ['negative','positive', 'neutral' ]
freq = [d0 , d1 , d2]
ax.bar(sentiment,freq)
plt.show()

'''