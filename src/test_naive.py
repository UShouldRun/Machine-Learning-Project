import math
import random
import pandas as pd
import numpy as np
import os
import sys

from naive import *


def k_fold(data, k=5, seed=70) -> float:

    X = data[:, :-1] 
    y = data[:, -1] 
    np.random.seed(seed)
    indices= np.arange(len(X))
    np.random.shuffle(indices)

    fold_size=len(X)//k 
    accs = []

    for i in range(k):
        start= i*fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X)

        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))

        info = MeanAndStdDevForClass(train_data)

        predictions = getPredictions(info, test_data)

        acc = accuracy_rate(test_data, predictions)

        accs.append(acc)


    return np.mean(accs)



def load_dataset(filename):
    
    try:
        full_path = os.path.join(folder_path, filename)
        df= pd.read_csv(full_path)
        
        for col in df.columns[:-1]:  
            df[col] = pd.to_numeric(df[col], errors='coerce')

        data = df.values.tolist()
        return data
    except Exception as e:
        print(f"Error loading dataset {filename}: {e}")
        return None

if __name__ == "__main__":
    
    match int(sys.argv[1]):
        case 1:
            folder_path = "../utils/multiclass_classification/"
        case 2:    
            folder_path="../utils/noise_outliers/"
        case 3:
            folder_path= "../utils/class_imbalance/"
    
    accuracies=[]

    file_list = [f for f in os.listdir(folder_path)]

    for filename in file_list:
        raw_data= load_dataset(filename)

        try:
            data = encode_class(raw_data)
            
            accuracy=k_fold(np.array(data))
            if 2<accuracy<98:
                accuracies.append(accuracy)
                print('Accuracy of the model:', accuracy)

        except Exception as e:
            raise e

    print("Accuracy ",np.mean(accuracies))
    print("Std deviation ",np.std(accuracies))
