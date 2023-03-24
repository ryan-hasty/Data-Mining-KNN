from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

xtrain = []
ytrain = []
xtest = []
ytest = []

def dataSplit(dataSet):
    for i in dataSet.traindataset:
        xtrain.append(i.values)
        ytrain.append(i.key)

    for i in dataSet.testdataset:
        xtest.append(i.values)
        ytest.append(i.key)

def SVMModel(kernel):
    kernel.fit(xtrain,ytrain)
    y_prediction = kernel.predict(xtest)
    accuracy = kernel.score(xtest, ytest)

    return accuracy

def DimReduction():
   pca = PCA(n_components=2)

   plotTrainData = pca.fit_transform(xtrain)
   plotTestData = pca.fit_transform(xtest)

   plt.scatter(plotTrainData[:, 0], plotTrainData[:, 1], label = 'training data')
   plt.scatter(plotTestData[:, 0], plotTestData[:, 1], label = 'testing data')
   plt.legend()
   plt.show()
