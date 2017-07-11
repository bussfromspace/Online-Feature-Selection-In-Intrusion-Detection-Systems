import pdb
import os, sys
import numpy as np
import pandas as pd

from io import StringIO
from learn import learnKbest
from visuals import PlotAndSave
from preprocess import Normalize, SplitandNormalize, ConcatenateandNormalize, getUnivariateStatistics

def main(argv):

	X = pd.DataFrame()
	Y = pd.DataFrame()

	X12 = pd.read_csv('packetStatistics12_9.txt')
	Y12 = pd.read_csv('labels12_9.txt', header=None)
	#X122 = pd.read_csv('packetStatistics12_8.txt')
	#Y122 = pd.read_csv('labels12_8.txt', header=None)
	#X123 = pd.read_csv('packetStatistics12_1.txt')
	#Y123 = pd.read_csv('labels12_1.txt', header=None)
	Xtrain12, Xtest12, Ytrain12, Ytest12 = SplitandNormalize(X12, Y12, 0.4)

	ranks12 = getUnivariateStatistics(Xtrain12, Ytrain12) 

	print ('June 12th:')
	print('fscore based feature extraction:')
	trainTime, testTime, precision, accuracy = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='fscore')
	PlotAndSave(trainTime, testTime, precision, accuracy, 'CompareFScore12.eps')
	print ('tree based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='tree-based')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareTreeBased12.eps')
	print('Chi2 feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='Chi2')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareChiSquare12.eps')
	print ('mutual info based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='mutualInfo')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareMutualInfo12.eps')
	print ('recursive feature elimination with LogReg:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='RFE')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareRecursiveRFE12.eps')	
	print ('Get Avg of univariate feature importances:')	
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain12, Ytrain12, Xtest12, Ytest12, ranks12, sortby='Mean')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareAvgImportance12.eps')	
	del X12, Y12
	del Xtrain12, Ytrain12, Xtest12, Ytest12

	X13 = pd.read_csv('packetStatistics13_9.txt')
	Y13 = pd.read_csv('labels13_9.txt', header=None)
	#X132 = pd.read_csv('packetStatistics13_8.txt')
	#Y132 = pd.read_csv('labels13_8.txt', header=None)
	#X133 = pd.read_csv('packetStatistics13_1.txt')
	#Y133 = pd.read_csv('labels13_1.txt', header=None)
	Xtrain13, Xtest13, Ytrain13, Ytest13 = SplitandNormalize(X13, Y13, 0.4)
	ranks13 = getUnivariateStatistics(Xtrain13, Ytrain13)

	print ('June 13th:')
	print('fscore based feature extraction:')
	trainTime, testTime, precision, accuracy = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='fscore')
	PlotAndSave(trainTime, testTime, precision, accuracy, 'CompareFScore13.eps')
	print ('tree based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='tree-based')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareTreeBased13.eps')
	print('Chi2 feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='Chi2')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareChiSquare13.eps')
	print ('mutual info based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='mutualInfo')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareMutualInfo13.eps')
	print ('recursive feature elimination with LogReg:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='RFE')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareRecursiveRFE13.eps')	
	print ('Get Avg of univariate feature importances:')	
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain13, Ytrain13, Xtest13, Ytest13, ranks13, sortby='Mean')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareAvgImportance13.eps')	
	del X13, Y13
	del Xtrain13, Ytrain13, Xtest13, Ytest13

	X14 = pd.read_csv('packetStatistics14_8.txt')
	Y14 = pd.read_csv('labels14_8.txt', header=None)

	Xtrain14, Xtest14, Ytrain14, Ytest14 = SplitandNormalize(X14, Y14, 0.4)
	ranks14 = getUnivariateStatistics(Xtrain14, Ytrain14)

	print ('June 14th:')
	print('fscore based feature extraction:')
	trainTime, testTime, precision, accuracy = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='fscore')
	PlotAndSave(trainTime, testTime, precision, accuracy, 'CompareFScore14.eps')
	print ('tree based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='tree-based')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareTreeBased14.eps')
	print('Chi2 feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='Chi2')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareChiSquare14.eps')
	print ('mutual info based feature extraction:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='mutualInfo')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareMutualInfo14.eps')
	print ('recursive feature elimination with LogReg:')
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='RFE')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareRecursiveRFE14.eps')	
	print ('Get Avg of univariate feature importances:')	
	trainTime, testTime, trainMSE, testMSE = learnKbest(Xtrain14, Ytrain14, Xtest14, Ytest14, ranks14, sortby='Mean')
	PlotAndSave(trainTime, testTime, trainMSE, testMSE, 'CompareAvgImportance13.eps')	
	del X14, Y14
	del Xtrain14, Ytrain14, Xtest14, Ytest14

	
	
if  __name__ == '__main__':
    main(sys.argv)
