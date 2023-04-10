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
# Define k values to be tested
    k_values = [1, 3, 7, 9]
    formatteddata = data.Get_CFV_Data()
    if(userinput == 1):
        for k_value_for_knn in k_values:
            #Grab the dataset input 1
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

            knn_accuracy_array = []
            knn_sensitivity_array = []
            knn_specificity_array = []
            knn_tp_array = []
            knn_fp_array = []
            knn_tn_array = []
            knn_fn_array = []

            svm_accuracy_array = []
            svm_sensitivity_array = []
            svm_specificity_array = []
            svm_tp_array = []
            svm_fp_array = []
            svm_tn_array = []
            svm_fn_array = []

            for k_value in range(5):
                # Cross-fold validation
                dataset = data.Five_CFV_Split(k_value, formatteddata[0], formatteddata[1])
                xtrain, ytrain, xtest, ytest = s.DataSplit(dataset.group.traindataset, dataset.group.testdataset)
                ypredict, accuracy, ytest = s.SVMModel(sigmoid, xtrain, ytrain, xtest, ytest)
                tn, fp, fn, tp = s.ConfusionMatrix(ytest, ypredict)
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)

                print("\nFOR FOLD =", k_value+1)
                print("SVM")
                print("Accuracy", accuracy)
                print("Sensitivity", sensitivity)
                print("Specificity", specificity)
                print("True Positive", tp)
                print("True Negative", tn)
                print("False Negative", fn)
                print("False Positive", fp)

                svm_overall_accuracy += accuracy
                svm_overall_true_positive += tp
                svm_overall_true_negative += tn
                svm_overall_false_negative += fn
                svm_overall_false_positive += fp

                svm_accuracy_array.append(accuracy)
                svm_sensitivity_array.append(sensitivity)
                svm_specificity_array.append(specificity)
                svm_tp_array.append(tp)
                svm_fp_array.append(fp)
                svm_tn_array.append(tn)
                svm_fn_array.append(fn)

                knn_metrics = [a.ModelMeasures(dataset.group, k.KNN(dataset.group, k_value_for_knn))]
                knn_accuracy, knn_sensitivity, knn_specificity, true_positive, true_negative, false_negative, false_positive = map(sum, zip(*knn_metrics))
                print("\nKNN")
                print("Accuracy:", knn_accuracy)
                print("Sensitivity:", knn_sensitivity)
                print("Specificity:", knn_specificity)
                print("True Positive:", true_positive)
                print("True Negative:", true_negative)
                print("False Negative:", false_negative)
                print("False Positive:", false_positive)

                knn_overall_accuracy += knn_accuracy
                knn_overall_sensitivity += knn_sensitivity
                knn_overall_specificity += knn_specificity

                knn_overall_true_positive += true_positive
                knn_overall_true_negative += true_negative
                knn_overall_false_negative += false_negative
                knn_overall_false_positive += false_positive

                knn_accuracy_array.append(knn_accuracy)
                knn_sensitivity_array.append(knn_sensitivity)
                knn_specificity_array.append(knn_specificity)

                knn_tp_array.append(true_positive)
                knn_fp_array.append(false_positive)
                knn_tn_array.append(true_negative)
                knn_fn_array.append(false_negative)


            print("\nOVERALL DATA:")
            print("SVM")
            print("Accuracy:", svm_overall_accuracy/5)
            print("True Positive:", svm_overall_true_positive/5)
            print("True Negative:", svm_overall_true_negative/5)
            print("False Negative:", svm_overall_false_negative/5)
            print("False Positive:", svm_overall_false_positive/5)

            print("KNN")
            print("Accuracy:", knn_overall_accuracy/5)
            print("Sensitivity:", knn_overall_sensitivity/5)
            print("Specificity:", knn_overall_specificity/5)
            print("True Positive:", knn_overall_true_positive/5)
            print("True Negative:", knn_overall_true_negative/5)
            print("False Negative:", knn_overall_false_negative/5)
            print("False Positive:", knn_overall_false_positive/5)

            a.PlotAcSpSe(knn_accuracy_array, knn_sensitivity_array, knn_specificity_array, "KNN, K value = " + str(k_value_for_knn))
            a.PlotAcSpSe(svm_accuracy_array, svm_sensitivity_array, svm_specificity_array, "SVM")
            a.PlotTpFpTnFn(knn_tp_array,knn_fp_array,knn_tn_array,knn_fn_array, "KNN, K value = " + str(k_value_for_knn))
            a.PlotTpFpTnFn(svm_tp_array,svm_fp_array,svm_tn_array,svm_fn_array, "SVM")



        



main()