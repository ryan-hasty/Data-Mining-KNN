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
import numpy as np
import data as d
from heapq import nlargest
from collections import Counter

#calculate data for jaccards_coeff
def pqr_count(dataSet):
    #for each test value 
    for i in range(len(d.test_value_key)):
        distances = []
        #compare against each train value 
        for j in range(len(d.train_value_key)):
            p_count = 0
            q_count = 0
            r_count = 0
            #find matching values per column
            for k in range(len(d.test_value_key[0])):
                if d.test_value_key[i][k] == 1 and d.train_value_key[j][k] == 1:
                    p_count += 1
                elif d.test_value_key[i][k] == 0 and d.train_value_key[j][k] == 1:
                    q_count += 1
                elif d.test_value_key[i][k] == 1 and d.train_value_key[j][k] == 0:
                    r_count += 1
            #Calculate JC
            jc = jaccards_coefficient(p_count, q_count, r_count)
            #Add to distances 
            distances.append(jc)
        #Append all distances to the corresponding dataset 
        dataSet.dataset[i].distances = distances


def jaccards_coefficient(p, q, r):
  """
  Calculates Jaccard's coefficient.
  Jaccard's coefficient is used to determine likeness, and has the
  following formula: `p / (r + q + r)`.
  :param int p: (# of variables positive for both)
  :param int q: (# of variables positive in Q not R)
  :param int r: (# of variables positive in R no Q)
  :return float:    the result of the calculation.
  """
  #If there are no differences, return 1 
  if (q+r) == 0:
    return 1
  else:
    jc = p/(r+q+r)
    return jc
  
#Find the nearest neighbors and return the most occuring classification as the new label 
def Neighbors(k, valueToPredict, trainingDataset ):
    value = list()
    newvalue = list()
    nn = nlargest(k, valueToPredict)
    for i in nn:
        value.append(valueToPredict.index(i))
    
    for j in value:
        newvalue.append(trainingDataset.dataset[j].key)
    most_frequent_value = max(set(newvalue), key=newvalue.count)

    return most_frequent_value

#def Prediction():

# ---------------------------------------------------------------------------- #
             
def main():
    trainDataSet = d.StructTrainingData()
    testDataSet = d.StructTestData()
    pqr_count(testDataSet)
    example = testDataSet.dataset

    
    for i in range(19):
        nn = Neighbors(1, example[i].distances, trainDataSet)
        print("Predicted class: ", nn)

    

    #for i in example:
        #print(i.distances)
        #print('\n')

main()