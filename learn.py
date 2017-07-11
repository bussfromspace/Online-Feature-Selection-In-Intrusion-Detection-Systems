import pdb
import time
import os, sys
import numpy as np
import pandas as pd

from io import StringIO
from preprocess import selectKbest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, mean_squared_error, classification_report, precision_score, accuracy_score

#names = ["K Neighbors", "Log Regression", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost", "Naive Bayes", "QDA"]
names = ["Log Regression", "Decision Tree"]
classifiers = [
	#KNeighborsClassifier(3),
	LogisticRegression(solver='lbfgs', C=1e5),
	DecisionTreeClassifier(max_depth=5)]
	#RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
	#MLPClassifier(activation='logistic',alpha=1),
	#AdaBoostClassifier(),
	#GaussianNB(),
	#QuadraticDiscriminantAnalysis()]

def learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='fscore'):

	maxSelection = 20
	#define metrics
	precision = np.array([]).reshape(0, maxSelection)
	accuracy  = np.array([]).reshape(0, maxSelection)
	trainO    = np.array([]).reshape(0, maxSelection)
	testO     = np.array([]).reshape(0, maxSelection) 

	#iterate over classifiers
	for name, clf in zip(names, classifiers):
		print (name)
		#define metrics
		precisionclf = np.empty(shape=[0])
		accuracyclf  = np.empty(shape=[0])
		trainOclf   = np.empty(shape=[0])
		testOclf    = np.empty(shape=[0]) 

		for kbest in range(1,maxSelection+1):
			print (kbest)
			#get selected features
			selectedFeatures = selectKbest(ranks, k=kbest, sortby=sortby)
			#fit the model using training data
			st = time.time()
			clf.fit(Xtrain[selectedFeatures], Ytrain.ravel())
			ft = time.time()
			Ypred = clf.predict(Xtest[selectedFeatures])
			fp = time.time()
			trainOclf = np.append(trainOclf, ft-st)
			testOclf  = np.append(testOclf,  fp-ft)
			precisionclf = np.append(precisionclf, precision_score(Ytest, Ypred, average='macro'))
			accuracyclf  = np.append(accuracyclf,  accuracy_score(Ytest, Ypred))
			if kbest%5 == 0:
				print ('\nClassification report:\n', classification_report(Ytest, Ypred))
				print ('\n confusion matrix:\n',  confusion_matrix(Ytest, Ypred))
		
		trainO = np.vstack([trainO, trainOclf]) 
		testO  = np.vstack([testO, testOclf])
		precision = np.vstack([precision, precisionclf])
		accuracy  = np.vstack([accuracy, accuracyclf])

		print ('\nprecision:\n', precisionclf)
		print ('\naccuracy:\n', accuracyclf)
		print ('\ntraining time:\n', trainOclf)
		print ('\ntesting time:\n', testOclf)
	trainO = pd.DataFrame(trainO.T, columns=names, index=selectedFeatures)
	testO  = pd.DataFrame(testO.T,  columns=names, index=selectedFeatures)
	precision = pd.DataFrame(precision.T, columns=names, index=selectedFeatures)
	accuracy  = pd.DataFrame(accuracy.T, columns=names, index=selectedFeatures)
	
	return trainO, testO, precision, accuracy 

