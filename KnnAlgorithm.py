# ---------------------------------------------------------------------------- #
# Author:     Ryan Hasty                                                      #
# Author:     Job McCully                                                      #
# Author:     Robert Summerlin                                                 #
# Author:     Nickolas Paternoster                                                         #
#                                                                              #
# Professor:  Dr. Chen                                                         #
# Class:      CSCI 4370 / CSCI 6397                                            #
# Assignment: Project 1                                                        #
# Date Last Modified:       25 February 2023                                                 #
# ---------------------------------------------------------------------------- #

import pandas as pd
from collections import defaultdict

# - Important code below ----------------------------------------------------- #
# Convert data frames to NumPy arrays
train = pd.read_csv("Training_dataset.csv")
trainL = train.values.tolist()
test = pd.read_csv("Testing_dataset.csv")
testL = test.values.tolist()

def jaccards_coefficient(p, q, r):
  """
  Parameters: int p, int q, int r
  -------
  :return float:    the result of the calculation.
  -------
  Jaccard's coefficient is used to determine likeness, and has the
  following formula: `p / (r + q + r)`.
  """
  
  if(p == 0):
      return 0
  else:    
      j_coefficient = p / (p + q + r)
  return j_coefficient
# ---------------------------------------------------------------------------- #

def getCoefficients(my_list):
    '''
    Parameters: List of items to be TESTED
    ----------
    my_list : A CSV file of rows/columns of wine data

    Returns: VOID
    -------
    This will just alter the lists accordingly and you can choose
    the highest percentages for jaccard's coefficient

    '''
    
    # Initialize counter/index variables
    countP = 0
    countQ = 0
    countR = 0
    zeros = 0
    coeff_list = []
    #idx = 0
    
    #for idx, x in enumerate(trainL):
        #print(idx, x)
        #for index, val in enumerate(x[2:]):
            #print(index, val)
    #print(my_list[:][2])
    for idx, element in enumerate(trainL):# For each row in training_dataset.csv
        # goes in here 200 times
        for index, value in enumerate(element[2:]):# For each attribute in the element
            #print("MY_LIST VALUE: " + str(my_list[:][index]) + " AT INDEX: " + str(index) + " ELEMENT[Index]: " + str(element[index]))
            if my_list[:][index] == 1 and element[index] == 1:# Count P (commonalities)
                if index <= 1:
                    continue
                else:
                    countP += 1
                    
            elif my_list[:][index]  == 0 and element[2:][index] == 1: # Count Q (differences)
                if index <= 1:
                    continue
                else:
                    countQ +=1
                    
            elif my_list[:][index] == 1 and element[index] == 0: # Count R (differences)
                if index <= 1:
                    continue
                else:
                    countR +=1
            else: # Increment indexes (Zeros is just how many where both wines have 0)
                zeros +=1
        
        # Replace line above with line below in order to see each wine name with score
        coeff_list.append((idx, jaccards_coefficient(countP,countQ,countR)))
        countP=countQ=countR=0
        
    return coeff_list.copy()
        

def get_knn(list_to_knn, k=1): # k's default value set to 1
    gotList = []
    
    # for each test row in the datasets
    for index, testRow in enumerate(list_to_knn.copy()):
        counter = 0
        # get the k number of highest coefficient per test element
        while counter < k:
            gotList.append([index, testRow[counter]])
            counter += 1
                
    return gotList
        
def prediction(knn_list):
    
    countPlus = 0 # 90+ counter
    countMinus = 0 # 90- counter

    d = defaultdict(list)
    
    # Transform list into dictionary for easier indexing
    for k, v in knn_list:
        d[k].append(v)
        
    print(d, "\n")
    
    # For every value in the dictionary test the value of the 90+, 90- on test
    for k in d:
        
        # RESET countPlus and countMinus for each index
        countPlus = 0
        countMinus = 0
        
        # For EVERY index in the dictionary, count the values associated with indexes
        for v in d[k]:
            if trainL[v[0]][1] == 1: 
                countPlus += 1 # If it's 1, add one to plus counter
            else:
                countMinus += 1 # If it's 0, add one to minus counter

        if countPlus > countMinus: # 90+ wins, assign 1
            d[k] = 1
        elif countPlus < countMinus:# 90- wins, assign 0
            d[k] = 0
        else:
            # TIE happens, where plus counter == minus counter, assign highest coefficient value
            # This is only code I'm not that sure about
            for v in d[k]:
                if trainL[v[0]][1] == 1: #and not v == 1 and not v == 0: # If index[0] value is equals 1 assign to 90+
                    d[k] = 1
                    continue
                else:
                    d[k] = 0 # assign to 90- if key, value != 1
                
    for k,v in d.items():
        print("Wine: ",k,"\nPrediction: ",v, "\n")
        
    return d

def calc_sensitivity(prediction_dict):
    
    actual = []
    true_positive = 0
    false_negative = 0
    
    # Obtain true calculations from testL classification attribute
    for x in testL:
        actual.append(x[1])
    
    # For index, value in the dictionary of values
    for k, v in enumerate(prediction_dict.values()):
        
        # If the prediction[i] == actual[i] then add 1 to true_positive
        if prediction_dict[k] == 1 and actual[k] == 1:
            print("Wine:",k," VALUED:", prediction_dict[k], "ACTUAL:", actual[k], "**True Positive**")
            true_positive += 1
            
        # If the prediction[i] is FALSE and actual[i] is Positive then add 1 to false_positive 
        elif prediction_dict[k] == 0 and actual[k] == 1:
            print("Wine:", k," VALUED:",prediction_dict[k], "ACTUAL:", actual[k], "**False Negative**")
            false_negative += 1
            
            # Otherwise, the wine is not getting counted
        else:
            print("Wine: ", k, "N/A")
    
    # Sensitivity = true_pos / (True_pos + false_neg)
    sensitivity = true_positive / (true_positive + false_negative)
    
    print("Sensitivity = ", true_positive, " / ", (true_positive + false_negative))
    print("Sensitivity: ", sensitivity, "\n")
    
    return sensitivity, true_positive, false_negative

def calc_specificity(prediction_dict):
    actual = []
    true_negative = 0
    false_positive = 0
    
    # Obtain true calculations from testL classification attrbute
    for x in testL:
        actual.append(x[1])
    
    # For index, value in the dictionary of values
    for k, v in enumerate(prediction_dict.values()):
        
        # If the prediction[i] == actual[i] then add 1 to true_negative
        if prediction_dict[k] == 0 and actual[k] == 0:
            print("Wine:",k," VALUED:", prediction_dict[k], "ACTUAL:", actual[k], "**True Negative**")
            true_negative += 1
            
        # If the prediction[i] is Positive and actual[i] is Negative then add 1 to false_positive 
        elif prediction_dict[k] == 1 and actual[k] == 0:
            print("Wine:", k," VALUED:",prediction_dict[k], "ACTUAL:", actual[k], "**False Positive**")
            false_positive += 1
            
            # Otherwise, the wine is not getting counted
        else:
            print("Wine: ", k, "N/A")
    
    # Sensitivity = true_negative / (false_pos + true_neg)
    specificity = true_negative / (false_positive + true_negative)
    print("Specificity = ", true_negative, " / ", (false_positive + true_negative))
    print("Specificity: ", specificity, "\n")
    
    return specificity, true_negative, false_positive
    

def calc_accuracy(true_p, true_n, false_p, false_n):
    #plug into formula
    accuracy = (true_p + true_n) / (true_p + true_n + false_p + false_n)
    return accuracy
    
def main():
    # Declare empty lists for later use
    full_list = []
    
    # Go through each row/column in test list/training list
    for element in testL:
        temporary = getCoefficients(element.copy()) # get coefficients
        full_list.append(temporary)
    
    # sort list by coefficient value
    for element in full_list:
        element.sort(key = lambda x: x[1], reverse = True)
        
    # K-Nearest-Neighbor indexes corresponding with index in training list
    knn_list = get_knn(full_list, 7) # ***INSERT K VALUE HERE***
    
    # Get predictions based on Jaccard's Coefficients    
    predictions = prediction(knn_list)
    
    # get sensitivity score
    sensitivity, true_p, false_n = calc_sensitivity(predictions)
    
    # get specificity
    specitivity, true_n, false_p = calc_specificity(predictions)
    
    # get accuracy
    accuracy = calc_accuracy(true_p, true_n, false_p, false_n)
    print("ACCURACY: ", accuracy)
    
main()