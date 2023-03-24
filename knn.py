from collections import Counter
import random

#done 
def jaccards_coefficient(p, q, r):
    
    if(p+q+r == 0): # Prevent division by 0
        return 0
    else:    
        j_coefficient = p / (p + q + r)
        
    return j_coefficient

def pqr_count(Dataset, k_value):
    for i in range(len(Dataset.testdataset)):
        distances = [None] * k_value
        key = [None] * k_value
        # compare against each train value
        for j in range(len(Dataset.traindataset)):
            p = 0
            q = 0
            r = 0
            # find matching values per column
            for k in range(len(Dataset.testdataset[i].values)):
                if Dataset.traindataset[j].values[k] == 1 and Dataset.testdataset[i].values[k] == 1:
                    p += 1
                elif Dataset.traindataset[j].values[k] == 0 and Dataset.testdataset[i].values[k] == 1:
                    q += 1
                elif Dataset.traindataset[j].values[k] == 1 and Dataset.testdataset[i].values[k] == 0:
                    r += 1
            # Calculate JC
            jc = jaccards_coefficient(p, q, r)
            # find the minimum distance and replace if current jc is larger
            if None in distances:
                min_index = distances.index(None)
                distances[min_index] = jc
                key[min_index] = Dataset.traindataset[j].key
            else:
                min_distance = min(distances)
                min_index = distances.index(min_distance)
                if jc > min_distance:
                    distances[min_index] = jc
                    key[min_index] = Dataset.traindataset[j].key

        new_key = key[0:k_value]
        Dataset.testdataset[i].prediction_keys = new_key




    
def predict(value_to_predict):
    counts = Counter(value_to_predict.prediction_keys)
    most_common = counts.most_common(1)
    most_common_value = most_common[0][0]
    return most_common_value
   
    
def knn(Dataset, k):
    predictions = list()
    pqr_count(Dataset, k)
    for i in Dataset.testdataset:
        predictions.append(predict(i))
    return predictions

