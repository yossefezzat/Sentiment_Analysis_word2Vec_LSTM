from gensim.models import Word2Vec
import pandas as pd
import re as re
import numpy as np
import random as random
import matplotlib.pyplot as plt

size_vec = 100

#read data from a file 'tweets'
def read_data(file_name = 'tweets.csv'):
    data = pd.read_csv(file_name)
    texts = data['text'].to_list()
    sentiment = data['airline_sentiment'].map({'positive': 1, 'negative': 0 , 'neutral': 2}).to_list()
    return texts , sentiment

#Data tokenization of tweets
def re_sub_data(data_text):
    sentences = []
    for text in data_text:
        tokens = re.sub(r"[^a-z0-9]+"," " , text.lower()).split()
        sentences.append(tokens)
    return sentences

#Run Word2Vec model in tweets data
def build_Word2Vec(sentences):
    model = Word2Vec(sentences , size= size_vec , window = 5 , min_count = 5 , workers= 4 , sg = 0 )
    return model   

#get vocab stored in word2Vec model
def get_vocab_list(model):
    vocab_list = np.array(list(model.wv.vocab.keys()))
    return vocab_list

#get average of words in certain sentence
def get_average_sentence(sentence , model):
    vocabs = get_vocab_list(model)
    words = np.intersect1d(sentence , vocabs)

    if words.shape[0] > 0:
        words_add = np.sum(model.wv[words] , axis=0)
        average = np.divide(words_add , float(len(sentence)))
        return average
    else:
        return np.zeros(size_vec).tolist()
    
#get average of all Sentences    
def get_average_sentences(all_sentences , model , sentiment):
    all_avg = list()
    index = 0 
    for sen in all_sentences:
        sen_avg = get_average_sentence(sen , model)
        all_avg.append( (sen_avg , sentiment[index]) )
        index += 1
    return all_avg

#get sum of words in a sentence
def get_sum_sentence(sentence , model):
    vocabs = get_vocab_list(model)
    words = np.intersect1d(sentence , vocabs)

    if words.shape[0] > 0:
        words_add = np.sum(model.wv[words] , axis=0)
        return words_add
    else:
        return np.zeros(size_vec).tolist()

#get sum of all sentences    
def get_sum_sentences(all_sentences , model , sentiment):
    all_sum = list()
    index = 0 
    for sen in all_sentences:
        sen_sum = get_sum_sentence(sen , model)
        all_sum.append( (sen_sum , sentiment[index]) )
        index += 1
    return all_sum

#dividing data to train and test
def divide_data(data):
    random.shuffle(data)    
    train_data = data[:int(len(data)*0.8)]
    test_data = data[int(len(train_data)):int(len(data)*1)]
    return train_data , test_data

#split data to data(x) and labels (y)
def split_data(X_Y_data):
    X = [row[0] for row in X_Y_data]
    Y = [row[1] for row in X_Y_data]
    return X , Y

#initiate Data to train and test after splitting
def init_data(data):
    train_data , test_data = divide_data(data)
    train_X , train_Y = split_data(train_data)
    test_X , test_Y = split_data(test_data)
    return train_X , train_Y , test_X , test_Y

#sorting Sentiment by 0,1,2
def sort_by_sentiment(sentences):
    sentences.sort(key = lambda x:x[1])
    return sentences

#Ploting statistics of Negative, Positive, Neutral 
def plot_data_statistcs(texts , sentiment):
    d0 = list(item for item in sentiment).count(0)
    d1 = list(item for item in sentiment).count(1)
    d2 = list(item for item in sentiment).count(2)
    objects = ('negative', 'positive', 'neutral')
    performance = [d0 , d1 , d2]
    plt.ylim([0 , d0+d1+d2])
    plt.bar(objects, performance, align='center', alpha=0.5)
    plt.xticks(objects, objects)
    plt.ylabel('data')
    plt.title('data statistics')
    plt.show()
    
texts , sentiment = read_data()    
plot_data_statistcs(texts , sentiment)