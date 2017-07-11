import pdb
import os, sys
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from io import StringIO

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "purple", "orange"]

def PlotAndSave(trainTime, testTime, precision, accuracy, filename):

	matplotlib.style.use('ggplot')
	plt.figure(figsize=(20,20))
	plt.subplot(2,2,1)
	plt.ylabel('traintime(sec)')
	trainTime.plot(ax=plt.gca(), marker='.', colormap='Paired', rot=45, markersize=15)

	plt.subplot(2,2,2)
	plt.ylabel('testtime(sec)')
	testTime.plot(ax=plt.gca(), marker='.', colormap='Paired', rot=45, markersize=15)

	plt.subplot(2,2,3)
	plt.ylim(0.5,1.0)
	plt.ylabel('precision')
	plt.xticks(range(len(precision.index)),precision.index)
	precision.plot(ax=plt.gca(),marker='.', colormap='Paired', rot=45, markersize=15)

	plt.subplot(2,2,4)
	plt.ylim(0.5,1.0)
	plt.ylabel('accuracy')
	plt.xticks(range(len(accuracy.index)),accuracy.index)
	accuracy.plot(ax=plt.gca(),marker='.', colormap='Paired', rot=45, markersize=15)

	plt.suptitle(filename)
	plt.savefig(filename, format='eps', dpi=600)
	#plt.show()
	plt.close()

