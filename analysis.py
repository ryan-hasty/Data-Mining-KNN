import matplotlib as plt

true_positive = 0
true_negative = 0
false_negative = 0
false_positive = 0

def positive_negative(DataSet, prediction):
    for i in range(len(DataSet.testdataset)):
        if(prediction[i] == 1 and DataSet.testdataset[i].key == 1):
            true_positive + 1
        elif(prediction[i] == 0 and DataSet.testdataset[i].key == 0):
            true_negative + 1
        elif(prediction[i] == 0 and DataSet.testdataset[i].key == 1):
            false_negative + 1
        elif(prediction[i] == 1 and DataSet.testdataset[i].key == 0):
            false_positive + 1

def get_sensitivity():
          
    if (true_positive + false_negative) != 0:    
        sensitivity = true_positive / (true_positive + false_negative)
    else:
        return 0
    
    return sensitivity

def get_specificity():
    if (false_positive + true_negative) != 0:    
        specificity = true_negative / (false_positive + true_negative)
    else:
        return 0
    
    return specificity

def get_accuracy(DataSet,prediction):

    true_positive = 0
    true_negative = 0
    false_negative = 0
    false_positive = 0

    for i in range(len(DataSet.testdataset)):
        if(prediction[i] == 1 and DataSet.testdataset[i].key == 1):
            true_positive += 1
        elif(prediction[i] == 0 and DataSet.testdataset[i].key == 0):
            true_negative += 1
        elif(prediction[i] == 0 and DataSet.testdataset[i].key == 1):
            false_negative += 1
        elif(prediction[i] == 1 and DataSet.testdataset[i].key == 0):
            false_positive += 1

    if(true_positive + true_negative + false_positive + false_negative != 0):
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) # Equation for accuracy
    else:
        accuracy = 0
        
    return accuracy


def plotModelPerformance(accuracy, plotspecificity, plotsensitivity):
   # Define xticks as a list comprehension
    xticks = [i for i in range(1, 21)]

# Plot the data and add labels
    plt.plot(accuracy, label='Accuracy')
    plt.plot(plotspecificity, label='Specificity')
    plt.plot(plotsensitivity, label='Sensitivity')

# Add legend and axis labels
    plt.legend()
    plt.xlabel('K Value')
    plt.ylabel('Accuracy, Specificity, Sensitivity')
    plt.xticks(xticks)
    plt.xlim(1, 20)
    plt.ylim(0.4, 1)

    # Show the plot
    plt.show()

    

def format_output(predicted_vals, sensitivity, specificity, accuracy, outF, k, test_answers):
    
    # Output specs to file
    outF.write("Real Grade\tPredicted Grade\n")
    for x in range(len(predicted_vals)):
        outF.write(str(int(test_answers[x])) + "\t\t\t" + str(int(predicted_vals[x])) + "\n")
    outF.write("\nK-value: " + str(k) + "\n")
    outF.write("Specificity: " + str(specificity)+ "\n")
    outF.write("Sensitivity: " + str(sensitivity)+ "\n")
    outF.write("Accuracy: " + str(accuracy) + "\n\n")

def output_file():
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