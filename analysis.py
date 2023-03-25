import matplotlib as plt

def ModelMeasures(data, predictions):
    true_positive = 0
    true_negative = 0 
    false_positive = 0 
    false_negative = 0

    for i in range(len(data.testdataset)):
        if predictions[i] == 1 and data.testdataset[i].key == 1:
            true_positive += 1
        elif predictions[i] == 0 and data.testdataset[i].key == 0:
            true_negative  += 1
        elif predictions[i] == 0 and data.testdataset[i].key == 1:
            false_negative  += 1
        elif predictions[i] == 1 and data.testdataset[i].key == 0:
            false_positive  += 1

    sensitivity = true_positive  / (true_positive  + false_negative ) if (true_positive  + false_negative ) != 0 else 0
    specificity = true_negative  / (false_positive  + true_negative ) if (false_positive  + true_negative ) != 0 else 0
    accuracy = (true_positive  + true_negative ) / (true_positive  + true_negative  + false_positive  + false_negative ) if (true_positive  + true_negative  + false_positive  + false_negative ) != 0 else 0
    
    return accuracy, sensitivity, specificity 