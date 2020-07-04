#Import Relevant Libraries
import numpy
import pandas
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

path = "student-mat.csv"
data = pandas.read_csv(path, sep=";")

target = "G3"#what we will predict
labels = numpy.array(data[target])#review lines.
features = numpy.array(data.drop([target], 1))
