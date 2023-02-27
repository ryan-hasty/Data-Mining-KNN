# ---------------------------------------------------------------------------- #
# Author:     Job McCully                                                      #
# Author:     Ryan Hasty                                                       #
# Author:     Robert Summerlin                                                 #
# Author:     Nickolas Paternoster                                             #
#                                                                              #
# Professor:  Dr. Chen                                                         #
# Class:      CSCI 4370 / CSCI 6397                                            #
# Assignment: Project 1                                                        #
# Date:       23 February 2023                                                 #
# ---------------------------------------------------------------------------- #

import pandas as pd

# - Important code below ----------------------------------------------------- #
# Import data from csv files 
training_dataset = pd.read_csv("Training_dataset.csv")
testing_dataset = pd.read_csv("Testing_dataset.csv")

#Name of wine 
train_type_key = training_dataset.iloc[:, 0].values.tolist()
test_type_key = testing_dataset.iloc[:, 0].values.tolist()
#Grade classification of wine 
train_class_key = training_dataset.iloc[:, 1].values.tolist()
test_class_key = testing_dataset.iloc[:, 1].values.tolist()
#Attributes 
train_value_key = training_dataset.iloc[:, 2:].values.tolist()
test_value_key = testing_dataset.iloc[:, 2:].values.tolist()

#Object to all values of a datapoint 
class DataPoint:
    type = ""
    key = 0
    values = list()
    distances = list()

#Object to store all datapoints 
class DataSet:
    traindataset = list()
    testdataset = list()

#Populate data object
def StructData(Data):

    for i in range(0,len(train_type_key)): 
        d = DataPoint()
        d.type = train_type_key[i]
        d.key = train_class_key[i]
        d.values = train_value_key[i]
        Data.traindataset.append(d)

    for i in range(len(test_type_key)): 
        d = DataPoint()
        d.type = test_type_key[i]
        d.key = test_class_key[i]
        d.values = test_value_key[i]
        Data.testdataset.append(d)

    return Data


def GetData():
    Data = DataSet()
    FormattedData = StructData(Data)
    return FormattedData



  