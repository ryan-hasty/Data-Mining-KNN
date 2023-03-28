import pandas as pd
import numpy as np

# - Important code below ----------------------------------------------------- #
# Import data from csv files 
training_dataset = pd.read_csv("Training_dataset.csv")
testing_dataset = pd.read_csv("Testing_dataset.csv")
full_dataset = pd.read_csv("Full Wine Data.csv")

# Name of wine 
train_type_key = training_dataset.iloc[:, 0].to_numpy()
test_type_key = testing_dataset.iloc[:, 0].to_numpy()
full_type_key = full_dataset.iloc[:, 0].to_numpy()

# Grade classification of wine 
train_class_key = training_dataset.iloc[:, 1].to_numpy()
test_class_key = testing_dataset.iloc[:, 1].to_numpy()
full_class_key = full_dataset.iloc[:, 1].to_numpy()

# Attributes 
train_value_key = training_dataset.iloc[:, 2:].to_numpy()
test_value_key = testing_dataset.iloc[:, 2:].to_numpy()
full_value_key = full_dataset.iloc[:, 2:].to_numpy()

# Object to store all values of a datapoint 
class DataPoint:
    def __init__(self):
        self.type = ""
        self.key = 0
        self.values = np.array([])
        self.prediction_keys = np.array([])

# Object to store all datapoints 
class DataSet:
    def __init__(self):
        self.traindataset = np.array([])
        self.testdataset = np.array([])

class SingleDataSet:
    def __init__(self):
        self.data = np.array([])

class CFVDataSets: 
    def __init__(self):
        self.group = SingleDataSet()

# Populate data object
def StructData(DataSet):
    # set empty array 
    traindataset = []
    testdataset = []

    # for the number of elements in the file, set the type, key, and values of each point and populate said point into traindataset. Traindataset is an array of objects. 
    for i in range(len(train_type_key)): 
        d = DataPoint()
        d.type = train_type_key[i]
        d.key = train_class_key[i]
        d.values = train_value_key[i]
        traindataset.append(d)
    # for the number of elements in the file, set the type, key, and values of each point and populate said point into testdataset. Testdataset is an array of objects. 
    for i in range(len(test_type_key)): 
        d = DataPoint()
        d.type = test_type_key[i]
        d.key = test_class_key[i]
        d.values = test_value_key[i]
        testdataset.append(d)

    #Sets the arrays within the DataSet object to the newly populated arrays 
    DataSet.traindataset = np.array(traindataset)
    DataSet.testdataset = np.array(testdataset)

    return DataSet


# Populate data object
def StructAllData(DataSet):
    # set empty array 
    dataset = []

    # for the number of elements in the file, set the type, key, and values of each point and populate said point into testdataset. Testdataset is an array of objects. 
    for i in range(len(full_type_key)): 
        d = DataPoint()
        d.type = full_type_key[i]
        d.key = full_class_key[i]
        d.values = full_value_key[i]
        dataset.append(d)

    #Sets the arrays within the DataSet object to the newly populated arrays 
    DataSet.data = np.array(dataset)

    return DataSet


def GetData():
    #Create new dataset object 
    Dataset = DataSet()
    #Format the data 
    FormattedData = StructData(Dataset)
    return FormattedData

def Get_CFV_Data():
    #Create new dataset object 
    Dataset = SingleDataSet()
    CFV = CFVDataSets()
    #Format the data 
    FormattedData = StructAllData(Dataset)
    np.random.shuffle(FormattedData.data)

    return FormattedData, CFV

def Five_CFV_Split(iteration, dataset,cfv):
# Split train and test datasets into 5 groups
    groups = np.array_split(dataset.data, 5)
    data = DataSet()

    for fold in range(5):
        if(fold == iteration):
            data.testdataset = groups[iteration]
        elif fold != iteration:
            data.traindataset = np.concatenate([data.traindataset, groups[fold]])
    cfv.group = data

    return cfv

