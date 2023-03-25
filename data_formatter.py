import pandas as pd
import numpy as np

# - Important code below ----------------------------------------------------- #
# Import data from csv files 
training_dataset = pd.read_csv("Training_dataset.csv")
testing_dataset = pd.read_csv("Testing_dataset.csv")

# Name of wine 
train_type_key = training_dataset.iloc[:, 0].to_numpy()
test_type_key = testing_dataset.iloc[:, 0].to_numpy()
generic_type_key = np.array([])

# Grade classification of wine 
train_class_key = training_dataset.iloc[:, 1].to_numpy()
test_class_key = testing_dataset.iloc[:, 1].to_numpy()
generic_class_key = np.array([])

# Attributes 
train_value_key = training_dataset.iloc[:, 2:].to_numpy()
test_value_key = testing_dataset.iloc[:, 2:].to_numpy()
generic_value_key = np.array([])

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

class CFVDataSets: 
    def __init__(self):
        self.group = DataSet()

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

def GetData():
    #Create new dataset object 
    Dataset = DataSet()
    #Format the data 
    FormattedData = StructData(Dataset)
    return FormattedData

def Get_CFV_Data():
    #Create new dataset object 
    Dataset = DataSet()
    CFV = CFVDataSets()
    #Format the data 
    FormattedData = StructData(Dataset)
    np.random.shuffle(FormattedData.traindataset)

    return FormattedData.traindataset, CFV

def Five_CFV_Split(iteration, dataset,cfv):
# Split train and test datasets into 5 groups
    groups = np.array_split(dataset, 5)
    data = DataSet()

    for fold in range(5):
        if(fold == iteration):
            data.testdataset = groups[iteration]
        elif fold != iteration:
            data.traindataset = np.concatenate([data.traindataset, groups[fold]])
    cfv.group = data

    return cfv

