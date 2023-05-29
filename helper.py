#!/usr/bin/env python

'''
This file contains some useful functions for file-managements, reading input data, etc. that can be used by other scripts
List of variables: 
CHROMOSOME_LIST: 1--22, X --> For CSREP, we only gets summary chromatin state maps for chromosomes 1--> 22, X, because not all our input samples have Y chromosome. 
NUM_BP_PER_BIN: 200 (bp)
NUM_BIN_PER_WINDOW: 50,000 (windows, each of length NUM_BP_PER_BIN)
NUM_BP_PER_WINDOW: NUM_BP_PER_BIN * NUM_BIN_PER_WINDOW

List of functions:
- make_dir(dir_path) --> create a directory (recursively) if it does not exist yet
- check_file_exist(fn) --> if not, exit the program
- create_folder_for_file(fn) --> usually used when fn is an output file path. This function will create the folder that contains the file fn
- check_dir_exist(dir_path) --> if not, exit the program
- get_command_line_integer(argument) --> try to convert to integer, if not succesfully then exit the program. This function is not entirely useful anymore given that we use argparse for all our scripts now. 
- get_list_from_line_seperated_file(fn) --> read in a file such that each line is an item in a list, using pandas series
- partition_file_list (file_list, num_cores) --> partition the file_list into a list of num_cores lists, as evely distributed as possible. Useful when we want to partition the list of outptu files into lists that can then be divided for different cores  to produce in parallel. 
'''
import os
import numpy as np
import pandas as pd
import h5py
from scipy import stats
from sklearn.metrics import mean_squared_error


CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X'] # we exclude chromosome Y because it's not available in all cell types


def obtain_training_SNPs(enformer_fn, gnomad_fn, feature_key):
	'''
	enformer_fn: fn of enformer data, h5 file
	gnomad_fn: data of MAF for different variants
	feature_key: SAD or SAR, since enformer provide features in SAD or SAR (log form of SAD)
	'''
	enf_f = h5py.File(enformer_fn, 'r') # keys: ['SAD', 'SAR', 'alt', 'chr', 'pos', 'ref', 'snp', 'target_ids', 'target_labels']
	enf_df = pd.DataFrame({'alt': enf_f['alt'], 'ref': enf_f['ref'], 'pos' : enf_f['pos'], 'snp' : enf_f['snp']})
	enf_df['alt'] = enf_df['alt'].apply(lambda x: x.decode('utf-8'))
	enf_df['ref'] = enf_df['ref'].apply(lambda x: x.decode('utf-8'))
	enf_df['snp'] = enf_df['snp'].apply(lambda x: x.decode('utf-8'))
	enf_df['full_id'] = enf_df['ref'] + '_' + enf_df['alt'] + '_' + enf_df['snp'] 
	enf_df.reset_index(inplace = True, drop = False) # column 'index' will correspond to the row index of SNP in enformer data 
	enf_df = enf_df[['full_id', 'index']]
	enf_df = enf_df.rename(columns = {'index': 'enf_idx'})
	# note: enf_df['pos'] corresponds to gno_df['end']
	gno_df = pd.read_csv(gnomad_fn, header = None, sep = '\t', index_col = None)
	gno_df.columns = ['chrom', 'start', 'end', 'rsID', 'ref', 'alt', 'maf'] 
	gno_df = gno_df[gno_df['rsID'] != '.'] # only pick the variants with clear rsID
	gno_df['full_id'] = gno_df['ref'] + '_' + gno_df['alt'] + '_' + gno_df['rsID']
	gno_df = gno_df.merge(enf_df, left_on = 'full_id', right_on = 'full_id', how = 'inner')
	gno_df['maf'] = gno_df['maf'].apply(lambda x: float(x[3:])) # from AF=7.12279e-03 to 0.007123
	gno_df['maf'] = np.log(gno_df['maf'] + 1e-5) # the threshold is so that we can avoid log of 0
	gno_df.sort_values(by = 'enf_idx', inplace = True, ignore_index = True)
	enf_input_X = (enf_f[feature_key])[list(gno_df['enf_idx'])]
	return gno_df, enf_input_X

def eval_yHat(y_hat, y, outPears_fn):
	'''
	calculate pearson correlation between predicted Y and the observed y
	'''
	res = stats.pearsonr(y_hat, y)
	mse = mean_squared_error(y, y_hat) # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
	df = pd.DataFrame(columns = ['pearsonr', 'pval', 'MSE'])
	df.loc[0] = [res[0], res[1], mse]
	df.to_csv(outPears_fn, header = True, index = False, sep = '\t')
	return


def remove_file(fn):
	try:
		os.remove(fn)
	except:
		print('File {} does not exist'.format(fn))
	return 
	
def make_dir(directory):
	try:
		os.makedirs(directory)
	except:
		print('Folder' + directory + ' is already created')



def check_file_exist(fn):
	if not os.path.isfile(fn):
		print("File: " + fn + " DOES NOT EXIST")
		exit(1)
	return True
	
def create_folder_for_file(fn):
	last_slash_index = fn.rfind('/')
	if last_slash_index != -1: # path contains folder
		make_dir(fn[:last_slash_index])
	return 

def check_dir_exist(dirName):
	if not os.path.isdir(dirName):
		print("Directory: " + dirName + " DOES NOT EXIT")
		exit(1)
	return

def get_command_line_integer(arg):
	try: 
		arg = int(arg)
		return arg
	except:
		print("Integer: " + str(arg) + " IS NOT VALID")
		exit(1)

def get_list_from_line_seperated_file(fn):
	# from a text file where each line contains an item, we will get a list of the items
	result =  list(pd.read_csv(fn, sep = '\n', header = None)[0]) # -->  a list with each entry being an element in a list. Note the [0] is necessary for us to get the first column
	return result

def partition_file_list(file_list, num_cores):
	results = [] # list of lists of file names
	num_files_per_core = int(np.around(len(file_list) / num_cores)) # round to the nearest decimal point, either up or down
	if num_files_per_core == 0:
		for file in file_list:
			results.append([file])
		for core_i in range(len(file_list), num_cores):
			results.append([])
		return results
	# else, the number of files per core is greater than 0, meaning only
	for core_i in range(num_cores):
		if core_i < (num_cores - 1):
			this_core_files = file_list[core_i * num_files_per_core : (core_i + 1) * num_files_per_core]
		elif core_i == (num_cores - 1):
			this_core_files = file_list[core_i * num_files_per_core :]
		results.append(this_core_files)
	return results
