import Sentiment_Analysis as data  #Old Data From Excel File 
from tensorflow import keras
import numpy as np  
from sklearn import metrics 

text = data.texts
sentiment = data.sentiment
sentences = data.re_sub_data(text)
model_data = data.build_Word2Vec(sentences)
average_data = data.get_sum_sentences( sentences , model_data , sentiment )
sum_data = data.get_sum_sentences( sentences , model_data , sentiment )

# changing label from LSTM from 3 to 1 answer to predict it
def change_label_to_1_class(data_Y):
    result = list()
    for dataa in data_Y:
        index = np.argmax(dataa)
        result.append(index)
    return result

# test model of LSTM 
def test_model(model ,test_X , test_Y):
    tested = model.predict(test_X , verbose=0)
    print(tested)
    tested = change_label_to_1_class(tested)
    accuracy = metrics.accuracy_score(test_Y, tested)
    print("Accuracy: %.2f%%" % (accuracy * 100))
    return accuracy

# prepare data for LSTM model  (Reshaping of train data and change y_train to_categorical) 
def prepare_model(average_or_sum):
    train_X , train_Y , test_X , test_Y = data.init_data(average_or_sum)
    train_X = np.array(train_X)
    train_X = train_X.reshape(train_X.shape[0], 1 , train_X.shape[1])
    train_Y = np.array(train_Y , dtype='int')
    train_Y = keras.utils.to_categorical( train_Y , num_classes = 3)
    test_X = np.array(test_X)
    test_X = test_X.reshape(test_X.shape[0], 1 , test_X.shape[1])
    return train_X, train_Y, test_X, test_Y


# Run all (Sum, Average, Extract Accuracy)
def LSTM_model(average_or_sum):
    train_X , train_Y , test_X , test_Y = prepare_model(average_or_sum)
    model = keras.Sequential()
    model.add(keras.layers.LSTM(120,input_shape=(1, data.size_vec)))
    model.add(keras.layers.Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' , metrics=['accuracy'])
    print(model.summary())
    history = model.fit(train_X , train_Y , epochs = 10 , batch_size = 64 ,  verbose=1)
    accuracy = test_model(model , test_X , test_Y)
    return accuracy , history.history['acc']












