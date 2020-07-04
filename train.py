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
path = "student-mat.csv"
data = pandas.read_csv(path, sep=";")
target = "G3" #what category we will predict

#eliminate = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 
#'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', ]
labels = numpy.array(data[target])#convert object to array
features = remStrArr(numpy.array(data.drop([target], 1)))
#data.drop() removes target from array. 
#next line splits data into training and test sections.
topModel = 0
topScore = 0
for i in range(100):
    featuresTrain, featuresTest, labelsTrain, labelsTest = split(features, labels, 0.1)
    model = linear_model.LinearRegression() #select linear regression model
    model.fit(featuresTrain, labelsTrain) #draw line of best fit using training data
    modelScore = model.score(featuresTest, labelsTest) #determine how well model performs.
    if (modelScore > topScore):
        topScore = modelScore
        topModel = model

file = open('topModel.pickle', 'wb')
pickle.dump(topModel, file)
file.close()
print("best score: " + str(topScore))
print("Time: " + str(t.monotonic()-start))
