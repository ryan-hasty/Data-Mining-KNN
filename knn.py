from collections import Counter

def JaccardsCoefficient(p, q, r):
    if p + q + r == 0:
        return 0
    else:
        return p / (p + q + r)

def PQR_Counter(Dataset, k_value):
    for i, test_data in enumerate(Dataset.testdataset):
        distances = [None] * k_value
        keys = [None] * k_value
        # compare against each train value
        for j, train_data in enumerate(Dataset.traindataset):
            # find matching values per column
            p, q, r = 0, 0, 0
            for k, (train_value, test_value) in enumerate(zip(train_data.values, test_data.values)):
                if train_value == 1 and test_value == 1:
                    p += 1
                elif train_value == 0 and test_value == 1:
                    q += 1
                elif train_value == 1 and test_value == 0:
                    r += 1
            # Calculate JC
            jc = JaccardsCoefficient(p, q, r)
            # find the minimum distance and replace if current jc is larger
            if None in distances:
                index = distances.index(None)
                distances[index] = jc
                keys[index] = train_data.key
            else:
                min_distance = min(distances)
                index = distances.index(min_distance)
                if jc > min_distance:
                    distances[index] = jc
                    keys[index] = train_data.key
        new_key = keys[:k_value]
        Dataset.testdataset[i].prediction_keys = new_key

def Predict(value_to_predict):
    counts = Counter(value_to_predict.prediction_keys)
    most_common = counts.most_common(1)
    most_common_value = most_common[0][0]
    return most_common_value

def KNN(Dataset, k):
    PQR_Counter(Dataset, k)
    predictions = [Predict(test_data) for test_data in Dataset.testdataset]
    return predictions
