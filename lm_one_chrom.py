import os
import pandas as pd
import numpy as np
import helper
import argparse 
from itertools import combinations
import helper 
from sklearn.linear_model import Ridge # start with Ridge, because it's faster
import pickle # for saving and loading the regression model
import h5py

def train_linear_regression(X, y, save_fn, alpha = 1.0):
	clf = Ridge(alpha=1.0)
	clf.fit(X, y) 
	print('Done training')
	#https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
	pickle.dump(clf, open(save_fn, 'wb'))
	print ('Done saving the model')
	return 

def lm_predict(X, model_fn):
	clf = pickle.load(open(model_fn, 'rb'))
	y_hat = clf.predict(X)
	return y_hat


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'This script will take in the data from file 12m Msh3-Q140 ATAC DA list.xlsx provided by Nan in their email on September 27 2022, and we will produce file with the same format as those provided by Peter, with columns chrom, start, end, padj, log2FC as the required columns')
	parser.add_argument('--enformer_fn', type = str, required = True, help = 'The .h5 files provided by Enformer')
	parser.add_argument('--gnomad_fn', type = str, required = True, help = 'gnomad file showing the MAF of variants')
	parser.add_argument('--model_fn', type = str, required= True, help = 'output file of the model parameters')
	parser.add_argument('--ALPHA', type = float, required= False, default = 1, help = 'Alpha values for Ridge regression')
	parser.add_argument('--eval_fn', type = str, required = False, help = 'where we will store the pearson correlation')
	args = parser.parse_args()
	helper.create_folder_for_file(args.model_fn)
	helper.check_file_exist(args.enformer_fn)
	helper.check_file_exist(args.gnomad_fn)
	print('Done getting command line arguments')
	if args.eval_fn == None:
		train_snp_df, train_X = helper.obtain_training_SNPs(args.enformer_fn, args.gnomad_fn)
		print('Done obtaining training data')
		train_linear_regression(train_X, train_snp_df['maf'], args.model_fn, alpha = 1.0)
	else:
		test_snp_df, test_X = helper.obtain_training_SNPs(args.enformer_fn, args.gnomad_fn)
		y_hat = lm_predict(test_X, args.model_fn)
		helper.eval_yHat(y_hat, test_snp_df['maf'], args.eval_fn)
	print('Done!')

