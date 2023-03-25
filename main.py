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
        dataset = data.Get_CFV_Data()
        overall_accuracy = 0
        overall_sensitivity = 0
        overall_specificity = 0
        count = 0
        for fold in dataset.group:
            knn_accuracy = 0
            knn_sensitivity = 0
            knn_specificity = 0
        # Split current fold into train and test sets
            s.DataSplit(fold)

        # Perform SVM classification and append accuracy results to list
            svm_accuracy.append(s.SVMModel(rbf))
            svm_accuracy.append(s.SVMModel(sigmoid))
            svm_accuracy.append(s.SVMModel(linear))
            svm_accuracy.append(s.SVMModel(poly))


        # Perform KNN classification for different k values and print metrics
            print("\nFOR FOLD  =", count+1)

            for k_value in range(20):
                predictions = k.KNN(fold, k_value+1)
                metrics = a.ModelMeasures(fold, predictions)
                knn_accuracy += metrics[0]
                knn_sensitivity += metrics[1]
                knn_specificity += metrics[2]


            print("Accuracy", knn_accuracy/20)
            print("Sensitivity", knn_sensitivity/20)
            print("Specificity", knn_specificity/20)
            knn_accuracy += metrics[0]
            knn_sensitivity += metrics[1]
            knn_specificity += metrics[2]
            count+=1
            overall_accuracy += knn_accuracy/20
            overall_sensitivity += knn_sensitivity/20
            overall_specificity += knn_specificity/20

        print("\nOVERALL DATA: ")
        print("Accuracy", overall_accuracy/5)
        print("Sensitivity", overall_sensitivity/5)
        print("Specificity", overall_specificity/5)


        



main()