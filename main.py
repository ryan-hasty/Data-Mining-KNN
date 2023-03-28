from sklearn import svm
import data_formatter as data
import svm as s
import knn as k
import analysis as a

#Variables 
svm_accuracy = list()
svm_confusionmatrix = list()
rbf = svm.SVC(kernel='rbf', C = 1)
poly = svm.SVC(kernel='poly', C = 1)
linear = svm.SVC(kernel='linear', C = 1)
sigmoid = svm.SVC(kernel='sigmoid', C = 1)


def main():

    userinput = int(input("Would you like to use Cross fold validation (1 or 0): "))

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
        knn_overall_accuracy = 0
        knn_overall_sensitivity = 0
        knn_overall_specificity = 0
        knn_overall_true_positive = 0
        knn_overall_true_negative = 0
        knn_overall_false_negative = 0 
        knn_overall_false_positive = 0
        svm_overall_accuracy = 0
        svm_overall_true_positive = 0
        svm_overall_true_negative = 0
        svm_overall_false_negative = 0 
        svm_overall_false_positive = 0


        for k_value in range(5):
            # Cross-fold validation
            dataset = data.Five_CFV_Split(k_value, formatteddata[0], formatteddata[1])

            xtrain, ytrain, xtest, ytest = s.DataSplit(dataset.group.traindataset, dataset.group.testdataset)
            ypredict, accuracy, ytest = s.SVMModel(sigmoid, xtrain, ytrain, xtest, ytest)
            tn, fp, fn, tp = s.ConfusionMatrix(ytest, ypredict)
            print("\nFOR FOLD =", k_value+1)
            print("SVM")
            print("Accuracy", accuracy)
            print("True Positive", tp)
            print("True Negative", tn)
            print("False Negative", fn)
            print("False Positive", fp)

            svm_overall_accuracy += accuracy
            svm_overall_true_positive += tp
            svm_overall_true_negative += tn
            svm_overall_false_negative += fn
            svm_overall_false_positive += fp

            knn_metrics = [a.ModelMeasures(dataset.group, k.KNN(dataset.group, k_value+1)) for k_value in range(5)]
            knn_accuracy, knn_sensitivity, knn_specificity, true_positive, true_negative, false_negative, false_positive = map(sum, zip(*knn_metrics))
            print("\nKNN")
            print("Accuracy:", knn_accuracy/5)
            print("Sensitivity:", knn_sensitivity/5)
            print("Specificity:", knn_specificity/5)
            print("True Positive:", true_positive/5)
            print("True Negative:", true_negative/5)
            print("False Negative:", false_negative/5)
            print("False Positive:", false_positive/5)

            knn_overall_accuracy += knn_accuracy
            knn_overall_sensitivity += knn_sensitivity
            knn_overall_specificity += knn_specificity
            knn_overall_true_positive += true_positive
            knn_overall_true_negative += true_negative
            knn_overall_false_negative += false_negative
            knn_overall_false_positive += false_positive

        print("\nOVERALL DATA:")
        print("SVM")
        print("Accuracy:", svm_overall_accuracy/5)
        print("True Positive:", svm_overall_true_positive/5)
        print("True Negative:", svm_overall_true_negative/5)
        print("False Negative:", svm_overall_false_negative/5)
        print("False Positive:", svm_overall_false_positive/5)

        print("KNN")
        print("Accuracy:", knn_overall_accuracy/25)
        print("Sensitivity:", knn_overall_sensitivity/25)
        print("Specificity:", knn_overall_specificity/25)
        print("True Positive:", knn_overall_true_positive/25)
        print("True Negative:", knn_overall_true_negative/25)
        print("False Negative:", knn_overall_false_negative/25)
        print("False Positive:", knn_overall_false_positive/25)



        



main()