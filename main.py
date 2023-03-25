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

    userinput = int(input("Crossfold validation or not (1 or 0): "))

    if(userinput == 0):
        #Grab the dataset input 
        dataset = data.GetData()

        s.DataSplit(dataset)
        svm_accuracy.append(s.SVMModel(rbf))
        svm_accuracy.append(s.SVMModel(sigmoid))
        svm_accuracy.append(s.SVMModel(linear))
        svm_accuracy.append(s.SVMModel(poly))
    

        for i in svm_accuracy: 
            print(i)
        for k_value in range(20):
            print("\nFOR K VALUE =", k_value+1)
            predictions = k.KNN(dataset, k_value+1)
            metrics = a.ModelMeasures(dataset, predictions)

            print("Accuracy", metrics[0])
            print("Sensitivity", metrics[1])
            print("Specificity", metrics[2])
    elif(userinput == 1):
        #Grab the dataset input 1
        formatteddata = data.Get_CFV_Data()
        overall_accuracy = 0
        overall_sensitivity = 0
        overall_specificity = 0
        overall_true_positive = 0
        overall_true_negative = 0
        overall_false_negative = 0 
        overall_false_positive = 0

        for k_value in range(5):
            # Cross-fold validation
            dataset = data.Five_CFV_Split(k_value, formatteddata[0], formatteddata[1])
            knn_metrics = [a.ModelMeasures(dataset.group, k.KNN(dataset.group, k_value+1)) for k_value in range(20)]
            knn_accuracy, knn_sensitivity, knn_specificity, true_positive, true_negative, false_negative, false_positive = map(sum, zip(*knn_metrics))

            print("\nFOR FOLD =", k_value+1)
            print("Accuracy:", knn_accuracy/20)
            print("Sensitivity:", knn_sensitivity/20)
            print("Specificity:", knn_specificity/20)
            print("True Positive:", true_positive/20)
            print("True Negative:", true_negative/20)
            print("False Negative:", false_negative/20)
            print("False Positive:", false_positive/20)

            overall_accuracy += knn_accuracy
            overall_sensitivity += knn_sensitivity
            overall_specificity += knn_specificity
            overall_true_positive += true_positive
            overall_true_negative += true_negative
            overall_false_negative += false_negative
            overall_false_positive += false_positive

        print("\nOVERALL DATA:")
        print("Accuracy:", overall_accuracy/100)
        print("Sensitivity:", overall_sensitivity/100)
        print("Specificity:", overall_specificity/100)
        print("True Positive:", overall_true_positive/100)
        print("True Negative:", overall_true_negative/100)
        print("False Negative:", overall_false_negative/100)
        print("False Positive:", overall_false_positive/100)



        



main()