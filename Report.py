
import matplotlib.pyplot as plt

import LSTM as lstm
import SVM_Model as svm



#svm_accuracy = svm.SVM_Model()

#average_accuracy = lstm.LSTM_model(lstm.average_data)

sum_accuracy = lstm.LSTM_model(lstm.sum_data)

'''
objects = ('SVM', 'Average Lstm', 'Sum Lstm')

performance = [svm_accuracy , average_accuracy , sum_accuracy]
plt.ylim([0 , 1])
plt.bar(objects, performance, align='center', alpha=0.5)
plt.xticks(objects, objects)
plt.ylabel('Accuracy')
plt.title('Test Accuracies')

plt.show()

'''