import sys, os
import json
import helper 
import argparse 

def rever_key_two_levels(data):
	'''
	if data = {'fcNN_params': {'input_dim': 5313}}
	then output {'input_dim' : 'fcNN_params'}
	'''
	reverse_keys = {} # keys: the secondary keys, values: the first keys associated with the seconday keys
	for main_key in data.keys():
		for sec_key in data[main_key].keys():
			reverse_keys[sec_key] = main_key
	return reverse_keys

def check_layers_params(data):
	'''
	when the number of layers from the list params_layers_sizes is different from the number of layers from params_layers_actFuncs
	It means users only wants to specify the size of each layers, and they ask us to apply default actFunc (relu) to all the layers
	So fix params_layers_actFuncs accordingly
	data: the dictionary object that we will dump to json file, for model config
	'''
	if len(data['fcNN_params']['params_layers_actFuncs']) != len(data['fcNN_params']['params_layers_sizes']):
		data['fcNN_params']['params_layers_actFuncs'] = ['relu'] * len(data['fcNN_params']['params_layers_sizes']) 
	return data

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'This script will generate the json files needed to creat model config files for training FCNN models')
	parser.add_argument('--default_fn', type = str, required = True, help = 'The default json file')
	parser.add_argument('--output_fn', type = str, required = True, help = 'output_fn')
	parser.add_argument('--params_layers_sizes', type = int, required = False, nargs='+', help = 'params_layers_sizes, such as [300,30,1]')
	parser.add_argument('--params_layers_actFuncs', type = str, required = False, nargs='+', choices = ['relu', 'sigmoid', 'tanh', 'elu', 'linear'], help = 'params_layers_actFuncs, such as ["relu", "relu", "relu"]')
	parser.add_argument('--dropout_proba', type = float, required= False, help = 'dropout_proba')
	parser.add_argument('--num_epoch', type = int, required= False, help = 'num_epoch')
	parser.add_argument('--learning_rate', type = float, required= False, help = 'learning_rate')
	parser.add_argument('--batch_size', type = int, required= False, help = 'batch_size')
	parser.add_argument('--annealing_warm_up', type = float, required= False, help = 'annealing_warm_up')
	parser.add_argument('--kl_latent_scale', type = float, required= False, help = 'kl_latent_scale')
	parser.add_argument('--kl_global_params_scale', type = float, required= False, help = 'kl_global_params_scale')
	parser.add_argument('--l2_regularization', type = float, required= False, help = 'l2_regularization')
	parser.add_argument('--use_validation_set', type = str, required= False, help = 'use_validation_set', choices = ['true', 'false']) # there are ways to get it to be boolean value, but I like it being string for now
	parser.add_argument('--validation_set_fract', type = float, required= False, help = 'validation_set_fract')
	parser.add_argument('--validation_freq', type = int, required= False, help = 'validation_freq')
	parser.add_argument('--log_training_freq', type = int, required= False, help = 'log_training_freq')
	parser.add_argument('--save_model_params_freq', type = int, required= False, help = 'save_model_params_freq')
	parser.add_argument('--num_batch_per_epoch', type = int, required= False, help = 'num_batch_per_epoch')
	args = parser.parse_args()
	helper.check_file_exist(args.default_fn)
	helper.create_folder_for_file(args.output_fn)
	print('Done getting command line arguments')
	# Load the existing JSON file
	data = json.load(open(args.default_fn))
	reverse_keys = rever_key_two_levels(data)
	# Get the flags that are not None from the users' input to this script
	flags = {k: v for k, v in vars(args).items() if v is not None}
	# Find the non-None flags, and manipulate the data as needed
	for flag, value in flags.items():
		if flag != 'default_fn' and flag != 'output_fn':
			data[reverse_keys[flag]][flag] = getattr(args, flag)
	# finally check that the users' input params_layers_sizes and params_layers_actFuncs have matching size (# layers)
	# if not, modify the users' input by default behavious (relu actFunc in all layers)
	data = check_layers_params(data)
	# Write the modified data to a new JSON file
	json.dump(data, open(args.output_fn, 'w'), indent=4)
