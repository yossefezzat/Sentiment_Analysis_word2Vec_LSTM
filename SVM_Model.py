import Sentiment_Analysis as data  #Old Data From Excel File 
import numpy as np
from sklearn import svm , metrics 


def train_model(data , train_X , train_Y):
    svm_model = svm.SVC(decision_function_shape='ovo')
    svm_model.fit(train_X, train_Y)
    return svm_model


def test_model(svm_model ,test_X , test_Y):
    tested = svm_model.predict(test_X)
    accuracy = metrics.accuracy_score(test_Y, tested)
    print("Accuracy: %.2f%%" % (accuracy * 100))
    return accuracy

def SVM_Model():
    text  = data.texts
    sentiment = data.sentiment
    sentences = data.re_sub_data(text)
    model = data.build_Word2Vec(sentences)
        
    average = data.get_average_sentences( sentences , model , sentiment )
    
    train_X , train_Y , test_X , test_Y = data.init_data(average)
    svm_model = train_model(average , train_X , train_Y )
    accuracy  = test_model(svm_model , test_X , test_Y  )
    return accuracy



#SVM_Model()