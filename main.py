from sklearn import svm
import data_formatter as data
import svm as s
import knn as k
import analysis as a

#Variables 
svm_accuracy = list()
rbf = svm.SVC(kernel='rbf', C = 1)
poly = svm.SVC(kernel='poly', C = 1)
linear = svm.SVC(kernel='linear', C = 1)
sigmoid = svm.SVC(kernel='sigmoid', C = 1)


def main():
    #Grab the dataset input 
    dataset = data.GetRandomizedData()

    s.DataSplit(dataset)
    svm_accuracy.append(s.SVMModel(rbf))
    svm_accuracy.append(s.SVMModel(sigmoid))
    svm_accuracy.append(s.SVMModel(linear))
    svm_accuracy.append(s.SVMModel(poly))
    

    for i in svm_accuracy: 
        print(i)
    for k_value in range(21):
        predictions = k.KNN(dataset, k_value+1)
        print(predictions)
        metrics = a.ModelMeasures(dataset, predictions)

        print("Accuracy", metrics[0])
        print("Sensitivity", metrics[1])
        print("Specificity", metrics[2])

main()