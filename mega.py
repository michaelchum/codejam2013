from sklearn.linear_model import SGDRegressor
from numpy import genfromtxt, savetxt
from numpy import shape
import numpy
import dateutil.parser
import math
import scipy as sp
import cPickle
#from sklearn import cross_validation
#from sklearn import datasets
#from sklearn import metrics
#from sklearn import linear_model
#from sklearn import svm
from sklearn import tree

def main():
    #clf = tree.DecisionTreeRegressor()
    with open('my_dumped_classifier.pkl', 'rb') as fid:
        clf = cPickle.load(fid)

    testset = genfromtxt(open('sampleinput.csv','r'), dtype=None, delimiter=',',usecols = (0,1,2,3,4))
    newtestset = numpy.zeros(testset.shape[0]*7).reshape(testset.shape[0],7)
    index1 = 0

    for date in testset:
        year = date[0][0:4]
        month = date[0][5:7]
        day = date[0][8:10]
        time = date[0][12:13] + date[0][14:16]
        float(month)
        float(day)
        float(time)
        float(date[1])
        float(date[3])
        newdata1 = [month,day,time,0,date[2],date[3],date[4]]
        newtestset[index1] = newdata1
        index1 += 1

    savetxt('newtestset.csv', newtestset, delimiter=',', fmt='%f')

    #predicted_probs = [[x[4]] for index, x in enumerate(clf.predict(test))]
    predicted_probs = clf.predict(newtestset)
    savetxt('output.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
    main()