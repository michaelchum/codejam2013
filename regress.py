#! /usr/bin/python
import numpy
import csv


convertfnc = lambda x: (float)(x[-11:-9]) + ((float)(x[-8:-6])/100.) - (float)(x[-4]) + ((float)(x[5:7])-1)*24.0
a = numpy.genfromtxt(open("data_set.csv",'rb'),skip_header=1,delimiter=",",converters={0:convertfnc},dtype=[None,float or None,None,None,float or None,None],usecols=(0,1,2,3,4,5))

def intro(x,y):
	start = x
	stop=x+4
	a[x+1][y] = .75*a[start][y]+0.25*a[stop][y]
	a[x+2][y] = 2.0/7.0*a[start][y] + 3.0/7.0*a[start+1][y] + 2.0/7.0*a[stop][y]
	a[x+3][y] = 1.0/9.0*a[start][y] + 2.0/9.0*a[start+1][y] + 3.0/9.0*a[start+2][y] + 3.0/9.0*a[stop][y]

def writeToFile():
	with open('intro_data2.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for i in range (0,len(a)):
			writer.writerow([a[i][0],a[i][1],a[i][2],a[i][3],a[i][4],a[i][5]])


x = 0
y = 4
z = 78070

for i in range(0, len(a)/5 + (78088-62469)/4):
	intro(x,1)
	intro(x,2)
	intro(x,3)
	intro(x,4)
	x+=y
	

for i in range (1,5):
	a[x+1][i] = (a[x][i]+a[x-1][i])/2
	a[x+2][i] = (a[x+1][i]+a[x][i])/2
	a[x+3][i] = (a[x+2][i]+a[x+1][i])/2

writeToFile()