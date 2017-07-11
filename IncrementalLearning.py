
import pdb
import os, sys
import numpy as np
import pandas as pd

from io import StringIO
from utils import convertIPs
from preprocess import Normalize
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import label_binarize, LabelBinarizer


def binarize(X, nclass):
	#binarize given data by checking number of the
	#classes given as nclasses
	assert nclass > 1, "One class can not be binarized, check classes within the data"
	lb = LabelBinarizer()
	xb = lb.fit_transform(X) 
	if nclass > 2:
		return lb, xb
	elif nclass == 2: 
		return lb, np.hstack((1-xb, xb))

def main(argv):

	#validation and predicted outputs 
	Yval  = np.empty(shape=[0, 1])
	Ypred = np.empty(shape=[0, 1])

	#Preprocess
	X = pd.read_csv('packetStatistics13_8.txt')
	Y  = pd.read_csv('labels13_8.txt', header=None)

	Y = np.asarray(Y[1])
	#convert IPs to 4 different features
	X = convertIPs(X)

	#drop columns with full NaN values
	X = X.drop(X.columns[X.apply(lambda col: pd.isnull(col).all() == True)], axis=1)
	for col_name in X.columns:
		col = X[col_name]
		col[col.isnull()] = int(-1)
	X = X.drop('Unnamed: 0',1)


	timeN = X['frame.time_epoch'][len(X)-1]
	time0 = X['frame.time_epoch'][0]
	window = 300 #time window in epochs
	skf = int((timeN - time0)/window)

	ind = (X['frame.time_epoch'] > time0) & (X['frame.time_epoch'] <= (time0 + window)) 
	X0, Y0 = X[ind], Y[np.where(ind)]

	X0 = Normalize(X0) 


	clf = SGDClassifier(loss="hinge", penalty="l2", average=True, warm_start=True)
	clf.fit(X0, Y0)
	lambda_ = 0.01 #regularization parameter
	mu_ = 0.2 #step size 

	nclass = len(np.unique(Y0))
	print (confusion_matrix(clf.predict(X0), Y0))


	#OFS truncation
	Wt = clf.coef_[0]
	sortind = np.argsort(abs(Wt))
	Wt[sortind[:27]] = 0
	clf.coef_[0] = Wt
	clf.fit(X0, Y0)
	print (confusion_matrix(clf.predict(X0), Y0))
	

	for i in range(2, skf+1): #skf+1
		print(i)
		#get the chunk of data in this time window
		ind = (X['frame.time_epoch'] > (time0 + (i-1)*window)) & (X['frame.time_epoch'] <= (time0 + (i)*window))
		Xi, Yi = X[ind], Y[np.where(ind)]
		Xi = Normalize(Xi, X0, 1.0) 

		pdb.set_trace()
		print (confusion_matrix(Yi, clf.predict(Xi)))		

		if len(np.unique(Yi)) > nclass:
			nclass = len(np.unique(Yi))
			del clf
			clf = SGDClassifier(loss="hinge", penalty="l2", warm_start=True)
			clf.fit(Xi, Yi)
			
		else:
			#clf.fit(Xi, Yi)
			#OFS truncation
			Wt = clf.coef_[0]
			sortind = np.argsort(abs(Wt))
			Wt[sortind[:27]] = 0
			clf.coef_[0] = Wt
			clf.fit(Xi, Yi)	

		
		Yval  = np.append(Yval, Yi)
		Ypred = np.append(Ypred, clf.predict(Xi))

	print (confusion_matrix(Yval, Ypred)) #Yval, Ypred


	
if  __name__ == '__main__':
    main(sys.argv)
