from sklearn.linear_model import SGDRegressor
from numpy import genfromtxt, savetxt
import numpy
import dateutil.parser

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('datasetinter.csv','r'), dtype=None, delimiter=',', usecols=(0,1,2,3,4,5,6,7)) 
    newdataset = numpy.zeros(dataset.shape[0]*7).reshape(dataset.shape[0],7)
    index = 0
    for date in dataset:
    	dateutil.parser.parse(date[0])
    	year = date[0][0:4]
    	month = date[0][5:7]
    	day = date[0][8:10]
    	time = date[0][12:13] + date[0][14:16]
    	newdata = [month,day,time,date[1],0,date[3],0,date[5]]
    	newdataset[index] = newdata
    	index += 1
    print newdataset

    testset = genfromtxt(open('sampleinput.csv','r'), dtype=None, delimiter=',',usecols = (0,1,2,3,4))
    print testset
    newtestset = numpy.zeros(testset.shape[0]*(5+2)).reshape(testset.shape[0],(5+2))
    index1 = 0

    for date in testset:
    	year = date[0][0:4]
    	month = date[0][5:7]
    	day = date[0][8:10]
    	time = date[0][12:13] + date[0][14:16]
    	newdata = [month,day,time,date[1],0,date[3],0]
    	newtestset[index1] = newdata
    	index1 += 1

    print newtestset

    target = [x[7] for x in newdataset]
    train = [x[0:7] for x in newdataset]
    
    clf = SGDRegressor(loss="huber")
    clf.fit(train, target)

    #predicted_probs = [[x[4]] for index, x in enumerate(clf.predict(test))]
    predicted_probs = clf.predict(newtestset)
    savetxt('output.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
    main()