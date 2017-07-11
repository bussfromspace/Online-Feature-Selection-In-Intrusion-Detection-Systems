import pdb
import os, sys
import numpy as np
import pandas as pd

from io import StringIO
from learn import learnKbest
from utils import convertIPs
from visuals import PlotAndSave
from preprocess import Normalize, Sanitize, SplitandNormalize, ConcatenateandNormalize, getUnivariateStatistics

def main(argv):

	X = pd.DataFrame()
	Y = pd.DataFrame()

	Xtrain = pd.read_csv('packetStatistics13_8.txt')
	Ytrain = pd.read_csv('labels13_8.txt', header=None)
	Xtest  = pd.read_csv('packetStatistics13_9.txt')
	Ytest  = pd.read_csv('labels13_9.txt', header=None)
	
	NXtrain = Normalize(convertIPs(Xtrain))
	Xtrain  = Normalize(Sanitize(Xtrain))
	
	Xtrain = Xtrain.drop('Unnamed: 0',1)
	Xtest  = Xtest.drop( 'Unnamed: 0',1)
	Xtrain = Xtrain.drop('tcp.flags.urg',1)
	Xtrain = Xtrain.drop('ip.flags',1)

	#Xtest  = Normalize(convertIPs(Xtest), avgWindow = NXtrain, avgSize = 0.4)
	Xtest  = Normalize(convertIPs(Xtest))
	Ytrain = Ytrain.values[:,1]
	Ytest  = Ytest.values[:,1]

	print('training and testing data have been received')
	ranks = getUnivariateStatistics(Xtrain, Ytrain) 
	
	#print ('June 12th:')
	print('fscore based feature extraction:')
	trainTime, testTime, precision, accuracy = learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='fscore')
	PlotAndSave(trainTime, testTime, precision, accuracy, 'FSCORE.eps')
	print ('tree based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='tree-based')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'TREEBASED.eps')
	print('Chi2 feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='Chi2')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CHI2.eps')
	#print ('mutual info based feature extraction:')
	#trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='mutualInfo')
	#PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'MUTUAL.eps')
	print ('recursive feature elimination with LogReg:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain, Ytrain, Xtest, Ytest, ranks, sortby='RFE')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'RFE.eps')		

	del Xtrain, Ytrain, Xtest, Ytest


	
	
if  __name__ == '__main__':
    main(sys.argv)
