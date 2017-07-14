
import pdb
import os, sys
import numpy as np
import pandas as pd

from io import StringIO
from utils import convertIPs
import matplotlib.pyplot as plt
from preprocess import Normalize
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report


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

	
def plot_coefficients(classifier, feature_names, top_features=20):
	if len(classifier.classes_) > 2:
		coef = classifier.coef_[2].ravel()
	else:
		coef = classifier.coef_[0].ravel()
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients =  np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	plt.figure(figsize=(15, 10))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
	plt.grid(True)
	plt.show()
	
	return top_coefficients

def main(argv):

	#validation and predicted outputs 
	Yval  = np.empty(shape=[0, 1])
	Ypred = np.empty(shape=[0, 1])

	
	#Preprocess
	X1 = pd.read_csv('packetStatistics-1.pcap.txt')
	Y1 = pd.read_csv('labels-1.pcap.txt', header=None)

	X2 = pd.read_csv('packetStatistics-2.pcap.txt')
	Y2 = pd.read_csv('labels-2.pcap.txt', header=None)

	#X3 = pd.read_csv('packetStatistics14_8.txt')
	#Y3 = pd.read_csv('labels14_8.txt', header=None)

	Y1 = np.asarray(Y1[1])
	#convert IPs to 4 different features
	X1 = convertIPs(X1)

	Y2 = np.asarray(Y2[1])
	#convert IPs to 4 different features
	X2 = convertIPs(X2)

	#Y3 = np.asarray(Y3[1])
	##convert IPs to 4 different features
	#X3 = convertIPs(X3)

	Y = np.append(Y1, Y2)
	X = X1.append(X2)

	#Y = np.append(Y,Y3)
	#X = X.append(X3)
	

	#drop columns with full NaN values
	X = X.drop(X.columns[X.apply(lambda col: pd.isnull(col).all() == True)], axis=1)
	for col_name in X.columns:
		col = X[col_name]
		col[col.isnull()] = int(-1)
	X = X.drop('Unnamed: 0',1)

	splitWindow= 50#100#500check
	N = int(len(X)/splitWindow)
	
	X0, Y0 = X.iloc[:N], Y[:N]
	X0 = Normalize(X0) 

	featureranker = SGDClassifier(loss="hinge", penalty="l2", average=True) #, warm_start=True
	featureranker.partial_fit(X0, Y0, np.unique(Y0))
	classVector = np.unique(Y0)
	nclass = len(np.unique(Y0))
	topN = 20

	classifier = SGDClassifier(loss="hinge", penalty="l2", average=True) #, warm_start=True

	if len(featureranker.classes_) > 2: #1 or 2 depending on the unknown class is coming or not
		coef = featureranker.coef_[2].ravel()
	else:
		coef = featureranker.coef_[0].ravel()
		top_positive_coefficients = np.argsort(coef)[-topN:]
		top_negative_coefficients = np.argsort(coef)[:topN]
		top_coefficients =  np.hstack([top_negative_coefficients, top_positive_coefficients])

	Xtest, Ytest = X.iloc[:N, top_coefficients], Y[:N]
	Xtest = Normalize(Xtest) 
	classifier.partial_fit(Xtest, Ytest, np.unique(Ytest))

	print(confusion_matrix(Y0, featureranker.predict(X0)))
	print(confusion_matrix(Ytest, classifier.predict(Xtest)))

	i=1
	j=2
	while j < splitWindow: #skf+1
		print (j)
		#get the data chunk
		Xnew, Ynew = X.iloc[i*N+1:j*N], Y[i*N+1:j*N]
		Xi = Normalize(Xnew, X0, 0.5) 
		Yi = Ynew


		Yval  = np.append(Yval, Yi)
		Ypred = np.append(Ypred, featureranker.predict(Xi))	

		# if classes change or not:
		if len(np.unique(Yi)) == nclass: 
			if (np.unique(Yi) == classVector).all():
				featureranker.partial_fit(Xi, Yi, np.unique(Yi))

			else:
				nclass = len(np.unique(Yi))
				classVector = np.unique(Yi)
				featureranker = SGDClassifier(loss="hinge", penalty="l2") #warm_start=True
				featureranker.partial_fit(Xi, Yi, np.unique(Yi))
		elif len(np.unique(Yi)) > 1:
			nclass = len(np.unique(Yi))
			classVector = np.unique(Yi)
			featureranker = SGDClassifier(loss="hinge", penalty="l2") #warm_start=True
			featureranker.partial_fit(Xi, Yi, np.unique(Yi))

		else: #if the coming chunk belongs to only one class
			i=i-1
		i=i+1
		j=j+1
		#save the last block as X0	
		X0 = Xnew
		Y0 = Ynew


	#visualise the top feature coefficients
	top_coefficients = plot_coefficients(featureranker, X.columns, topN)

	#write detection performance to console
	print ('\nClassification report:\n', classification_report(Yval, Ypred))
	print ('\n confusion matrix:\n',  confusion_matrix(Yval, Ypred))
	

if  __name__ == '__main__':
    main(sys.argv)
