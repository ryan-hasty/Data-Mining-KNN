import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

def DataSplit(traindataset, testdataset):
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []

    for i in traindataset:
        xtrain.append(i.values)
        ytrain.append(i.key)

    for i in testdataset:
        xtest.append(i.values)
        ytest.append(i.key)

    return xtrain, ytrain, xtest, ytest

def SVMModel(kernel, xtrain, ytrain, xtest, ytest):
    kernel.fit(xtrain, ytrain)
    ypredict = kernel.predict(xtest)
    accuracy = kernel.score(xtest, ytest)

    return ypredict, accuracy, ytest

def DimReduction(xtrain, xtest):
    pca = PCA(n_components=2)
    xall = xtrain + xtest
    plotData = pca.fit_transform(xall)

    plt.scatter(plotData[:len(xtrain), 0], plotData[:len(xtrain), 1], label = 'training data')
    plt.scatter(plotData[len(xtrain):, 0], plotData[len(xtrain):, 1], label = 'testing data')
    plt.legend()
    plt.show()

def ConfusionMatrix(ytest, ypredict):
    tn, fp, fn, tp = confusion_matrix(ytest, ypredict).ravel()
    return tn, fp, fn, tp

