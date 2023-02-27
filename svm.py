from sklearn import svm
import data as d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

xtrain = list()
ytrain = list() 
xtest = list()
ytest = list()
plotTrainData = list()
plotTestData = list()
clf = svm.SVC(kernel='rbf', C = 1)

def dataSplit(dataSet):
    for i in dataSet.traindataset:
        xtrain.append(i.values)
        ytrain.append(i.key)

    for i in dataSet.testdataset:
        xtest.append(i.values)
        ytest.append(i.key)


def SVMModel():
    clf.fit(xtrain,ytrain)
    y_prediction = clf.predict(xtest)
    accuracy = clf.score(xtest, ytest)

    return accuracy

def DimReduction():
   pca = PCA(n_components=2)

   plotTrainData = pca.fit_transform(xtrain, ytrain)
   plotTestData = pca.fit_transform(xtest, ytest)

   plt.scatter(plotTrainData[:, 0], plotTrainData[:, 1], label = 'training data')
   plt.scatter(plotTestData[:, 0], plotTestData[:, 1], label = 'testing data')
   plt.legend()
   plt.show()


def main():
    dataSet = d.GetData()
    dataSplit(dataSet)
    accuracy = SVMModel()
    DimReduction()
    print(accuracy)

main()
