import os, sys
import numpy as np
import pandas as pd

import pdb

from io import StringIO
from utils import convertIPs
from sklearn.linear_model import Lasso
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import  chi2, f_classif, mutual_info_classif, RFE

eps = np.finfo(float).eps

def Normalize(df, avgWindow = None, avgSize = 0):
	if avgSize > 0:
		df2 = avgWindow.iloc[int(len(avgWindow)*(1-avgSize)):,:]
		lendf = len(df)
		frames = [df, df2]
		result = pd.concat(frames)
		#replace min value of the data frame to NaN to have correct mean and variance calculations
		nanval = min(result.min(numeric_only=True))
		result = result.replace(nanval, np.nan)
		#result = result.drop(result.columns[result.apply(lambda col: col.var(skipna=True) == 0)], axis=1)
		dfnormalized = (result - result.min(skipna=True))/(result.max(skipna=True) - result.min(skipna=True))
		dfnormalized = dfnormalized.replace(np.nan, -1.0)
		dfnormalized = dfnormalized.iloc[:lendf,:]
	else:
		#replace min value of the data frame to NaN to have correct mean and variance calculations
		nanval = min(df.min(numeric_only=True))
		df = df.replace(nanval, np.nan)
		#df = df.drop(df.columns[df.apply(lambda col: col.var(skipna=True) == 0)], axis=1)
		dfnormalized = (df - df.min(skipna=True))/(df.max(skipna=True) - df.min(skipna=True))
		dfnormalized = dfnormalized.replace(np.nan, -1.0)
	return dfnormalized

def Sanitize(df):

	#convert IPs to 4 different features
	df = convertIPs(df)
	#drop columns with full NaN values
	df = df.drop(df.columns[df.apply(lambda col: pd.isnull(col).all() == True)], axis=1)
	for col_name in df.columns:
		col = df[col_name]
		col[col.isnull()] = int(-1)
	#drop columns with zero variance, i.e. constant in field 
	df = df.drop(df.columns[df.apply(lambda col: col.var(skipna=True) == 0)], axis=1)
	#SanitizedDF = df.replace(np.nan, -1.0)
	return df

def SplitandNormalize(X, Y, testsize):
	#drop the first columns since they are ranks repeated
	X = X.drop(X.columns[[0]], axis=1)
	Y = Y.drop(Y.columns[[0]], axis=1)

	X = Normalize(Sanitize(X))
	#split the data into training and test parts
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, np.asarray(Y), stratify= np.asarray(Y), test_size=testsize, random_state=42)
	Xtrain.index = range(len(Xtrain))
	Xtest.index = range(len(Xtest))
	return Xtrain, Xtest, Ytrain, Ytest

def ConcatenateandNormalize(X1, X2, X3, Y1, Y2, Y3, testsize):
	#drop the first columns since they are ranks repeated
	X1 = X1.drop(X1.columns[[0]], axis=1)
	Y1 = Y1.drop(Y1.columns[[0]], axis=1)
	X2 = X2.drop(X2.columns[[0]], axis=1)
	Y2 = Y2.drop(Y2.columns[[0]], axis=1)
	X3 = X3.drop(X3.columns[[0]], axis=1)
	Y3 = Y3.drop(Y3.columns[[0]], axis=1)
	X = pd.concat([X1, X2, X3])
	Y = np.vstack((Y1, Y2, Y3))

	X.index = range(len(X))
	X = Normalize(Sanitize(X))
	#split the data into training and test parts
	Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, np.asarray(Y), stratify= np.asarray(Y), test_size=testsize, random_state=42)
	Xtrain.index = range(len(Xtrain))
	Xtest.index = range(len(Xtest))
	return Xtrain, Xtest, Ytrain, Ytest

def getUnivariateStatistics(X, Y):
	#univariate F-test classification statistics
	ftest, _ = f_classif(X, Y.ravel())
	ftest /= (np.max(ftest)+eps)
	
	#random forest feature importances
	#rf = RandomForestClassifier()
	#rf.fit(X, Y.ravel())
	#rf = rf.feature_importances_/(np.max(rf.feature_importances_)+eps)

	#lasso features
	#lasso  = Lasso(alpha=.05)
	#lasso.fit(X, Y.ravel())
	#lassoCo = abs(lasso.coef_)/(np.max(abs(lasso.coef_))+eps)
	
	#chi2 for each feature
	#since chi2 takes only positive values for input, change nan values 
	Xtemp = X.replace(-1, 100)
	chi2score = chi2(Xtemp,Y)[0]
	chi2score = chi2score/(np.max(chi2score)+eps)

	#mutual information
	mi = mutual_info_classif(X, Y.ravel())
	mi /= np.max(mi)

	#tree based feature selection
	model = ExtraTreesClassifier()
	model.fit(X, Y.ravel())
	tf = (model.feature_importances_)/(np.max(model.feature_importances_)+eps)

	#RFE recursive feature selection
	estimator = LogisticRegression()
	selector = RFE(estimator, 1, step=1)
	selector = selector.fit(X, Y.ravel())
	rfe = selector.ranking_
	rfe = 1 - (rfe/max(rfe))

	ranks = pd.DataFrame((np.c_[ftest, tf, chi2score, mi, rfe]), index = X.columns, 
					columns=['fscore', 'tree-based', 'Chi2', 'mutualInfo', 'RFE'])
	ranks['Mean'] = np.mean(ranks,axis=1)
	return ranks

def selectKbest(ranks, k=10, sortby='fscore'):

	ranks = ranks.sort_values(by=sortby, ascending = False)
	#selecting
	selected = ranks[:k]
	return selected.index
	
 

