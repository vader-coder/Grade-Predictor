#Import Relevant Libraries
import numpy
import pandas
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time as t

def split (x, y, defaultTestSize):
    return sklearn.model_selection.train_test_split(x, y, test_size=defaultTestSize)

def remStrArr (data):
    length = len(data[0])#numpy.size(data, axis=0)
    i = 0
    while (i<length):#loop through columns
        if (isinstance(data[0, i], str)):#if type in row 0, column i is string.
            data = numpy.delete(data, i, axis=1);#drop that column
            length -= 1
            i -= 1
        i+=1
    return data

start = t.monotonic()
target = "G3"
model = pickle.load(open("topModel.pickle", "rb"))

path = "student-mat.csv"
data = pandas.read_csv(path, sep=";")

labels = numpy.array(data[target])#convert object to array
features = remStrArr(numpy.array(data.drop([target], 1)))
featuresTrain, featuresTest, labelsTrain, labelsTest = split(features, labels, 0.1)

predictions = model.predict(featuresTest)
print(predictions)
print(str(t.monotonic() - start) + " seconds")