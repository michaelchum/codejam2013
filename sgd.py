from sklearn.linear_model import SGDRegressor
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('sampledataset.csv','r'), delimiter=',', dtype='f8', usecols = (1,2,3,4,5))[1:] 
    print dataset
    target = [x[4] for x in dataset]
    train = [[x[0],x[1],x[2],x[3]]for x in dataset]
    test = genfromtxt(open('sampleinput.csv','r'), delimiter=',', dtype='f8', usecols = (1,2,3,4))
    clf = SGDRegressor(loss="squared_loss")
    clf.fit(train, target)
    #predicted_probs = [[x[4]] for index, x in enumerate(clf.predict(test))]
    predicted_probs = clf.predict(test)
    savetxt('output.csv', predicted_probs, delimiter=',', fmt='%f')

if __name__=="__main__":
    main()