import csv
import numpy as np

def get_features():
    return [ 
        'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 
        'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 
        'Hue', 'OD280/OD315 of diluted wines', 'Proline'
    ] 

def load_cw2_data(file = 'wine.data.csv'):

    with open(file, 'r') as f:
        data = np.asarray([
            [float(f) for f in d]
            for d in csv.reader(f, delimiter=' ')
        ])

    split = data[:, 0]
    labels = data[:, 1]
    data = data[:, 2:]

    train_set = data[split == 1]
    test_set = data[split == 2]
    train_labels = labels[split == 1]
    test_labels = labels[split == 2]

    return train_set, test_set, train_labels, test_labels