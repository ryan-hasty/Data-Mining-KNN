# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:41:24 2023

@author: RRhas
"""

import pandas as pd
import numpy as np
import matplotlib as plt # currently unused
import warnings

# Keep below ignore options just in case we need later on
#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) # ignore
#np.set_printoptions(suppress=True) # get rid of weird numpy formatting


# - Important code below ----------------------------------------------------- #
# Testing Dataset - Contains np array to work with as well as correct answers
test = pd.read_csv("Testing_dataset.csv")
test_data = test.to_numpy()
test_np = test_data[:,2:] # datset to work with
test_answers = test_data[:,1] # correct answers for test dataset


# Training Dataset - Contains np array to work with as well as the correct answers
train = pd.read_csv("Training_dataset.csv")
train_data = train.to_numpy()
train_np = train_data[:,2:] # dataset to work with
train_answers = train_data[:,1] # correct answers for training dataset

def get_jaccards(p, q, r):
    
    if(p == 0):
        return 0
    else:    
        j_coefficient = p / (p + q + r)
        
    return j_coefficient

def get_coeffs(train_arr, test_arr):
    
    # Can reassign these whenever we get new datasets
    train_elements = 19
    test_elements = 200
    
    my_coefficients = np.array([]) # all coefficients per testing row
    
    # Initialize counter variables
    count_p = 0
    count_q = 0
    count_r = 0
    
    for idx, train_row in enumerate(train_arr):
        for index, test_row in enumerate(test_arr):
            count_p = np.sum((train_row == 1) & (test_row == 1)) # Count common elements
            count_q = np.sum((train_row == 1) & (test_row == 0)) # Count 1/0 combo
            count_r = np.sum((train_row == 0) & (test_row == 1)) # Count 0/1 combo
            my_coefficients = np.append(my_coefficients, get_jaccards(count_p, count_q, count_r))
            count_p = count_q = count_r = 0
    
    # Round coefficient values by 4 decimal places
    my_coefficients = np.around(my_coefficients, 4)
    
    # Reshape to get right dimensions
    # testing_dataset count, training_dataset count
    my_coefficients = np.reshape(my_coefficients, (train_elements, test_elements)) # 19 tst, 200 trn
    
    return my_coefficients

def get_knns(arr_to_knn, k=1): # k's default value set to 1
    
    # Sort the array
    sorted_coeffs = np.apply_along_axis(np.sort, 1, arr_to_knn)[:, ::-1]
    
    # Create a new array to store the original and sorted indices
    index_arr = np.zeros_like(arr_to_knn, dtype=np.int32) # currently holds 0 here
    
    for i, row in enumerate(arr_to_knn):
        # Get the indices of the k largest values in the current row
        sorted_indices = np.argsort(row)[::-1][:k]
        # Store the original indices and their sorted indices in the new array
        index_arr[i, 0] = i
        index_arr[i, 1:k+1] = sorted_indices
    
    return sorted_coeffs, index_arr, k
    
def get_predictions(index_of_knns, k_val):
    
    # Initialize variables
    temp_arr = index_of_knns[:,1:k_val+1]
    vals_predicted = np.array([])
                            
    # For every row of coefficient index's
    for item in temp_arr:
        
        # Initialize counter variables to reset with each row being predicted
        classify_plus = 0
        classify_minus = 0
        
        # For every value in each row
        for index_val in item:
            if train_answers[index_val] == 1: # Count 90+
                classify_plus +=1
            elif train_answers[index_val] == 0: # Count 90-
                classify_minus +=1
            else:
                print("Something is wrong")
        
        if classify_plus > classify_minus: # Classified as 90+
            vals_predicted = np.append(vals_predicted, 1)
            
        elif classify_plus < classify_minus: # Classified as 90-
            vals_predicted = np.append(vals_predicted, 0)        
            
        else:
            if train_answers[item[0]] == 1: # Classifies as highest coeff class
                vals_predicted = np.append(vals_predicted, 1)
            else:
                vals_predicted = np.append(vals_predicted, 0)
                
    return vals_predicted


def get_sensitivity(predicted_vals):
          
    # True Positves / False Negatives
    tp = 0
    fn = 0
    
    # Compare predicted values with true values of test data
    for x in range(len(predicted_vals)):
        if predicted_vals[x] == 1 and test_answers[x] == 1:
            tp += 1
        elif predicted_vals[x] == 0 and test_answers[x] == 1:
            fn += 1
            
    # Sensitivity = true_pos / (True_pos + false_neg)
    if (tp + fn) != 0:    
        sensitivity = tp / (tp + fn)
    else:
        return 0, tp, fn
    
    return sensitivity, tp, fn

def get_specificity(predicted_vals):
    
    # True Negatives / False Positives
    tn = 0
    fp = 0
    
    # Compare predicted values with true values of test data
    for x in range(len(predicted_vals)):
        if predicted_vals[x] == 0 and test_answers[x] == 0:
            tn += 1
        elif predicted_vals[x] == 1 and test_answers[x] == 0:
            fp += 1
    
    # Specificity = true_negative / (false_pos + true_neg)
    specificity = tn / (fp + tn)
    
    return specificity, tn, fp

def get_accuracy(tp, tn, fp, fn):

    if(tp + tn + fp + fn != 0):
        accuracy = (tp + tn) / (tp + tn + fp + fn) # Equation for accuracy
    else:
        accuracy = 0
        
    return accuracy

def format_output(predicted_vals, sensitivity, specificity, accuracy, outF, k):
    
    # Output specs to file
    outF.write("Real Grade\tPredicted Grade\n")
    for x in range(len(predicted_vals)):
        outF.write(str(int(test_answers[x])) + "\t\t\t" + str(int(predicted_vals[x])) + "\n")
    outF.write("\nK-value: " + str(k) + "\n")
    outF.write("Specificity: " + str(specificity)+ "\n")
    outF.write("Sensitivity: " + str(sensitivity)+ "\n")
    outF.write("Accuracy: " + str(accuracy) + "\n\n")
    
    
def main():
    
    # Create/Overwrite file
    output_file = open("knn_out.txt", "w")
    output_file.write("TEAM 7 RESULT REPORT \n\n")
    
    # Assign k here
    for k in range(1,21):
        
        # Get coefficients of every train/test pair
        j_coeffs = get_coeffs(train_np, test_np)
        
        # Get k-nearest neigbors of every train_row
        coeff_of_knns, index_of_knns, k_val = get_knns(j_coeffs, k)
        
        # Get predictions based on knn indexes
        predicted_vals = get_predictions(index_of_knns, k_val)
        
        # Sensitivity
        sensitivity, tp, fn = get_sensitivity(predicted_vals)
        
        # Specificity
        specificity, tn, fp = get_specificity(predicted_vals)
        
        # Accuracy
        accuracy = get_accuracy(tp, tn, fp, fn)
        
        # Print out the variables and output to txt doc
        format_output(predicted_vals, sensitivity, specificity, accuracy, output_file, k_val)
        
    # Close file
    output_file.close()
        
    # Open file again to make sure that it is correct
    file_to_open = open("knn_out.txt", "r")
    print(file_to_open.read())

main()