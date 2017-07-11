
import os, sys
import numpy as np
import pandas as pd
from io import StringIO

def convertIPs(df):
	header = df.columns
	#check if the features input has source and destination address
	if any(header == 'ip.src'):
		df = df.join(df['ip.src'].str.split('.', expand=True))
		df = df.rename(columns={0: 'saddr4', 1: 'saddr3', 2: 'saddr2', 3:'saddr1'})
		del df['ip.src']
	if any(header == 'ip.dst'):
		df = df.join(df['ip.dst'].str.split('.', expand=True))
		df = df.rename(columns={0: 'daddr4', 1: 'daddr3', 2: 'daddr2', 3:'daddr1'})
		del df['ip.dst']
	df = df.apply(pd.to_numeric, errors='coerce')
	return df

def convertFromHex(df):
	if (df[:2] == '0x'):
		dti = df[2:]
		dti = int(dti,16)
	else:
		dti = int(df, 16)
	return dti

def convertFromStr(df):
	if df == 'GET':
		df = int(1)
	elif df == 'HEAD':
		df = int(2)
	elif df == 'POST':
		df = int(3)
	elif df == 'PUT':
		df = int(4)
	elif df == 'DELETE':
		df = int(5)
	elif df == 'CONNECT':
		df = int(6)
	elif df == 'OPTIONS':
		df = int(7)
	elif df == 'TRACE':
		df = int(8)
	elif df == 'PATCH':
		df = int(9)
	else:
		df = int(-1)
	return df

def convertHexString(df):
	df['eth.type'] = df.apply(lambda  row: convertFromHex(row['eth.type']), axis=1)
	df['ip.id']    = df.apply(lambda  row: convertFromHex(row['ip.id']), axis=1)
	df['ip.flags'] = df.apply(lambda  row: convertFromHex(row['ip.flags']), axis=1)
	df['ip.checksum'] = df.apply(lambda  row: convertFromHex(row['ip.checksum']), axis=1)
	return df

def convertMACString(df):
	df = df.join(df['eth.src'].str.split(':', expand=True))
	df = df.rename(columns={0: 'SMC1',1:'SMC2',2:'SMC3',3:'SDC1',4:'SDC2',5:'SDC3'})
	del df['eth.src']
	df['SMC1'] = df.apply(lambda  row: convertFromHex(row['SMC1']), axis=1)
	df['SMC2'] = df.apply(lambda  row: convertFromHex(row['SMC2']), axis=1)
	df['SMC3'] = df.apply(lambda  row: convertFromHex(row['SMC3']), axis=1)
	df['SDC1'] = df.apply(lambda  row: convertFromHex(row['SDC1']), axis=1)
	df['SDC2'] = df.apply(lambda  row: convertFromHex(row['SDC2']), axis=1)
	df['SDC3'] = df.apply(lambda  row: convertFromHex(row['SDC3']), axis=1)
	
	df = df.join(df['eth.dst'].str.split(':', expand=True))
	df = df.rename(columns={0: 'DMC1',1:'DMC2',2:'DMC3',3:'DDC1',4:'DDC2',5:'DDC3'})
	del df['eth.dst']
	df['DMC1'] = df.apply(lambda  row: convertFromHex(row['DMC1']), axis=1)
	df['DMC2'] = df.apply(lambda  row: convertFromHex(row['DMC2']), axis=1)
	df['DMC3'] = df.apply(lambda  row: convertFromHex(row['DMC3']), axis=1)
	df['DDC1'] = df.apply(lambda  row: convertFromHex(row['DDC1']), axis=1)
	df['DDC2'] = df.apply(lambda  row: convertFromHex(row['DDC2']), axis=1)
	df['DDC3'] = df.apply(lambda  row: convertFromHex(row['DDC3']), axis=1)
	return df

def convertHTTPMethods(df):
	df['http.request.method'] = df.apply(lambda  row: convertFromStr(row['http.request.method']), axis=1)
	return df


