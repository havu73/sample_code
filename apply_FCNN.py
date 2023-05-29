import sys, os
import json
import pandas as pd
import numpy as np
import helper
import argparse 
from itertools import combinations
import helper 
import h5py
from scipy import stats
import random
import torch
# these two objects are from FCNN.py
from FCNN import FCNN
from FCNN import Data_w_Target

# ref: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 9999) -> None:
	np.random.seed(seed)
	random.seed(seed)
	## pytorch
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# Set a fixed value for the hash seed
	os.environ["PYTHONHASHSEED"] = str(seed)
	print(f"Random seed set as {seed}")

def train_model(enformer_fn, gnomad_fn, model_folder, model_params_fn, random_seed, enformer_data_mode):
	data = Data_w_Target(enformer_fn, gnomad_fn, enformer_data_mode) # this class contains the data, declared in FCNN.py file
	model_params = json.load(open(model_params_fn))
	print('Done reading in model parameters')
	model = FCNN(model_params, random_seed, model_folder)
	model.train_model(data)
	return 

def FCNN_predict(enformer_fn, gnomad_fn, model_folder, model_params_fn, random_seed, enformer_data_mode):
	data = Data_w_Target(enformer_fn, gnomad_fn, enformer_data_mode) # this class contains the data, declared in FCNN.py file
	model_params = json.load(open(model_params_fn))
	print('Done reading in model parameters')
	model_fn = os.path.join(model_folder, 'train.model')
	validation_fn = os.path.join(model_folder, 'validation.model')
	if os.path.isfile(validation_fn):
		model_fn = validation_fn
	model = FCNN(model_params, random_seed, model_folder)
	model.load_pretrained_model(model_fn)
	try:
		batch_size = model.training_params['batch_size']
	except:
		batch_size = 256
	y_hat = model.predict_MAF(data, batch_size).cpu().detach().numpy()
	y_target = data.get_target_pred().cpu().detach().numpy()
	return y_hat, y_target



def toy_instance(model_params_fn, random_seed, model_folder):
	num_samples = 10000
	input_dim = 5313
	output_dim = 1
	data = helper.ToyDataset(num_samples, input_dim, output_dim) # defined in helper.py
	model_params = json.load(open(model_params_fn))
	model_params['fcNN_params']['input_dim'] = input_dim
	model_params['fcNN_params']['params_layers_sizes'] = [2600, 1300, 650, 325, 160, 80,1]
	model_params['fcNN_params']['params_layers_actFuncs'] = ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu']
	print('Done reading in model parameters')
	model = FCNN(model_params, random_seed, model_folder)
	model.train_model(data)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'This script take in data from enformer and gnomad, and then train a fully connected neural network model to predict the minor allel frequency based on feature data from enformer. This script can also evaluate the pearson correlation of predicted MAF vs. observed MAF, given a trained model')
	parser.add_argument('--enformer_fn', type = str, required = True, help = 'The .h5 files provided by Enformer')
	parser.add_argument('--gnomad_fn', type = str, required = True, help = 'gnomad file showing the MAF of variants')
	parser.add_argument('--model_folder', type = str, required= True, help = 'output folder of the model parameters')
	parser.add_argument('--model_params_fn', type = str, required= True, help = 'json file specifying the model architecture')
	parser.add_argument('--eval_fn', type = str, required = False, help = 'If this is not provided, the code will assume that user want to train the model using data provided from --enformer_fn and --gnomad_fn first. If this file is provided from user, then the code will assume we want to do model evaluation, and it will save the pearson correlation, MSE in this file.')
	parser.add_argument('--random_seed', type = int, required = False, default = 9999, help = 'random_seed')
	parser.add_argument('--enformer_data_mode', type = str, required = False, default = 'SAD', choices = ['SAD', 'SAR'], help = 'the dataset within enformer_fn that we will take to be input feature')
	args = parser.parse_args()
	helper.make_dir(args.model_folder)
	helper.check_file_exist(args.enformer_fn)
	helper.check_file_exist(args.gnomad_fn)
	helper.check_file_exist(args.model_params_fn)
	set_seed(args.random_seed)
	print('Done getting command line arguments')
	# toy_instance(args.model_params_fn, args.random_seed, args.model_folder)
	if args.eval_fn != None: # evaluate the model
		helper.create_folder_for_file(args.eval_fn)
		y_hat, y_target = FCNN_predict(args.enformer_fn, args.gnomad_fn, args.model_folder, args.model_params_fn, args.random_seed, args.enformer_data_mode)
		helper.eval_yHat(y_hat, y_target, args.eval_fn)
	else: # train the model, 
		train_model(args.enformer_fn, args.gnomad_fn, args.model_folder, args.model_params_fn, args.random_seed, args.enformer_data_mode)
	
	print('Done!')
