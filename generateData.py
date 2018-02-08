import numpy as np

labels = ('B', 'M') 

features = [
            ('radius', 5, 1), 
            ('perimeter', 10, 2)
           ]

def featureMean(feature):
    return feature[1]

def featureStdDev(feature):
    return feature[2]

def generateValue(feature):
    return np.random.normal(featureMean(feature), featureStdDev(feature))

def generateData(rows):
    data = []
    for i in range(rows):
        row = [labels[randBool()]] 
        for f in features:
            row.append(generateValue(f))
        data.append(row)
    return data

def randBool():
    return np.random.randint(0,1)
