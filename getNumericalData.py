import pandas
import numpy

#eventually we will make this put the numerical data in a .csv file by itself so we can load it.

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

labels = numpy.array(data[target])#convert object to array
features = remStrArr(numpy.array(data.drop([target], 1)))

