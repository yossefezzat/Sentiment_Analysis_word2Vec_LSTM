import matplotlib.pyplot as plt
import LSTM as lstm
import SVM_Model as svm


svm_accuracy = svm.SVM_Model()
average_accuracy = lstm.LSTM_model(lstm.average_data)
sum_accuracy = lstm.LSTM_model(lstm.sum_data)

# plotting models accuracies
def plotting(objects , limits , performance, label , title):
    objects = objects
    performance = performance
    plt.ylim(limits)
    plt.bar(objects, performance, align='center', alpha=0.5)
    plt.xticks(objects, objects)
    plt.ylabel(label)
    plt.title(title)
    plt.show()

test_accuracy_objects = ('SVM', 'Average Lstm', 'Sum Lstm')
test_accuracy_performance = [svm_accuracy , average_accuracy , sum_accuracy]
limit = [0 , 1]


plotting(test_accuracy_objects, limit, test_accuracy_performance , 'Accuracy' ,'Test Accuracies')