# ---------------------------------------------------------------------------- #
# Author:     Job McCully                                                      #
# Author:     Ryan Hasty                                                       #
# Author:     Robert Summerlin                                                 #
# Author:     ???                                                              #
#                                                                              #
# Professor:  Dr. Chen                                                         #
# Class:      CSCI 4370 / CSCI 6397                                            #
# Assignment: Project 1                                                        #
# Date:       23 February 2023                                                 #
# ---------------------------------------------------------------------------- #

import pandas as pd
import numpy as np

# - Important code below ----------------------------------------------------- #
# Convert data frames to NumPy arrays
train = pd.read_csv("Training_dataset.csv")
trainL = train.values.tolist()
test = pd.read_csv("Testing_dataset.csv")
testL = test.values.tolist()

blows = []

def jaccards_coefficient(p, q, r):
  """
  Calculates Jaccard's coefficient.
  
  Jaccard's coefficient is used to determine likeness, and has the
  following formula: `p / (r + q + r)`.

  :param int p: TODO, describe.
  :param int q: TODO, describe.
  :param int r: TODO, describe.
  :return float:    the result of the calculation.
  """
  
  if(p == 0):
      return 0
  else:    
      j_coefficient = p / (p + q + r)
  return j_coefficient
# ---------------------------------------------------------------------------- #

def testit(my_list):
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
    
    for element in trainL:# For each row in training_dataset.csv
        # goes in here 200 times
        for value in range(len(element)):# For each attribute in the element
            if my_list[value] == 1 and element[value] == 1: # Count P (commonalities)
                countP += 1
            elif my_list[value]  == 0 and element[value] == 1: # Count Q (differences)
                countQ +=1
            elif my_list[value] == 1 and element[value] == 0: # Count R (differences)
                countR +=1
            else: # Increment indexes (Zeros is just how many nulls)
                zeros +=1
        
        # testing things
        #coeff_list.append(jaccards_coefficient(countP,countQ,countR))
        
        #coeff_list.append([idx,jaccards_coefficient(countP,countQ,countR)])
        #idx +=1
        
        # Replace line above with line below in order to see each wine name with score
        coeff_list.append([element[0],jaccards_coefficient(countP,countQ,countR)])
        countP=countQ=countR=0
    return coeff_list.copy()
        

def get_knn(list_to_knn, k=1):
    print("help")
    
    """ Im stuck super bad here and I need a nap """
        
        
        
def main():
    wine_list = [] # Declare empty list for later use
    
    # Use line below if you want to see original binary values for each test wine
    #print(testL)
    
    for element in testL: # Go through each element and add it
        temporary = testit(element.copy())
        wine_list.append(temporary.copy())
        # Use below code if you want to see the wine names associated with the indexes
        #wine_list.append([element[0],temper.copy()])
        
    for i in range(len(wine_list)):
        wine_list[:][:][i].sort(key=lambda x: x[:][1:][0],reverse=True)
    
    
    """
    
    3-D array of test_list_row_wine_name[train_list_row[subscript[0] = name of train_wine,
    subscript[1+] = Coefficients for each test_row/train_row]]
    
    """
    np_wine_list = np.array(wine_list)
    print(np_wine_list)
    #get_knn(wine_list, k=2)

main()