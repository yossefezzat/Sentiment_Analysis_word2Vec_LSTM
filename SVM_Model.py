import Sentiment_Analysis as data
from sklearn import svm , metrics 


# Training SVM model 
def train_model(data , train_X , train_Y):
    svm_model = svm.SVC(decision_function_shape='ovo')
    svm_model.fit(train_X, train_Y)
    return svm_model

# test prediction with test data to get test accuracy 
def test_model(svm_model ,test_X , test_Y):
    tested = svm_model.predict(test_X)
    accuracy = metrics.accuracy_score(test_Y, tested)
    print("Accuracy: %.2f%%" % (accuracy * 100))
    return accuracy

# model prediction test 
def predict_sentence(sentence, model, svm_model):
    arrSentence = [sentence]
    sentences = data.re_sub_data(arrSentence)
    average = data.get_average_sentence(sentences, model)
    result  = svm_model.predict([average])
    if(result == 0):
        answer = 'negative'
    elif(result == 1):
        answer = 'positive'
    else:
        answer = 'neutral' 
    return answer


# Run all (model and accuracy of test and prediction )
def SVM_Model():
    text  = data.texts
    sentiment = data.sentiment
    sentences = data.re_sub_data(text)
    model = data.build_Word2Vec(sentences)
        
    average = data.get_average_sentences( sentences , model , sentiment )
    
    train_X , train_Y , test_X , test_Y = data.init_data(average)
    svm_model = train_model(average , train_X , train_Y )
    accuracy  = test_model(svm_model , test_X , test_Y  )
    print(predict_sentence('i like it very much' , model , svm_model))
    return accuracy


