import matplotlib.pyplot as plt

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
    
    return accuracy, sensitivity, specificity, true_positive, true_negative, false_negative, false_positive


def PlotAcSpSe(accuracy_scores, sensitivity_scores, specificity_scores, title):
    plt.plot(range(1, 6), accuracy_scores, marker='o', label='Accuracy')
    plt.plot(range(1, 6), sensitivity_scores, marker='o', label='Sensitivity')
    plt.plot(range(1, 6), specificity_scores, marker='o', label='Specificity')
    plt.xlabel('Fold')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.xticks(range(1, 6))
    plt.show()

def PlotTpFpTnFn(tp,fp,tn,fn, title):
    plt.plot(range(1, 6), tp, marker='o', label='True Positive')
    plt.plot(range(1, 6), fp, marker='o', label='False Positive')
    plt.plot(range(1, 6), tn, marker='o', label='True Negative')
    plt.plot(range(1, 6), fn, marker='o', label='False Negative')
    plt.xlabel('Fold')
    plt.ylabel('Number')
    plt.title(title)
    plt.legend()
    plt.xticks(range(1, 6))
    plt.show()