import sys, os
import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler # for sampling the data for training and validation separately
from torch.utils.data import Dataset # for declaring ToyDataset
import helper 
import time
import tqdm

class ToyDataset(Dataset): 
	# code taken from ChatGPT
	'''
	This class is redundant for the model training, but I keep them because it was handy in debugging the model
	It just creates toy data sets with user-defined dimensions, all input values are generated randomly
	'''
	def __init__(self, num_samples, input_dim, output_dim):
		self.num_samples = num_samples
		self.input_dim = input_dim
		self.output_dim = output_dim

		# Generate random data
		self.data = torch.randn(num_samples, input_dim).float()
		self.targets = torch.randn(num_samples).float()

	def __len__(self):
		return self.num_samples

	def __getitem__(self, index):
		sample = self.data[index]
		target = self.targets[index]
		return sample, target

class Data_w_Target:
	def __init__(self, enformer_fn:str, gnomad_fn: str, enformer_data_mode: str='SAD'):
		gno_df, enf_input_X = helper.obtain_training_SNPs(enformer_fn, gnomad_fn, enformer_data_mode) # this function is shared across all the models (FCNN, different regressions, etc.)
		self.enf_input_X = torch.tensor(enf_input_X).float()
		self.maf = torch.tensor(gno_df['maf']).float()
		self.sample_size = len(gno_df)
		print('Done reading in input data from GNOMAD and Enformer')

	def __getitem__(self, index):
		X = self.enf_input_X[index, :]
		Y = self.maf[index]
		return X, Y

	def __len__(self):
		return self.sample_size

	def get_target_pred(self):
		return self.maf

class FCNN(nn.Module):
	"""
	Fully connected neural network with user-specified dimension and activating functions
	"""
	def __init__(self, params, random_seed, model_folder):
		"""
		Some input parameters that specifies model's architecture:
		- input_dim: (Int) # of experiments from Enformer 
		- params_layers_sizes: (List) List of sizes of FCNN linear layers
		- nonlinear_activation: (Str) Type of non-linear activation to apply on each hidden layer
		- dropout_proba: (Float) Dropout probability applied on all hidden layers. If 0.0 then no dropout applied
		"""
		super().__init__()
		self.random_seed = random_seed
		self.dtype = torch.float32
		# First we will init components of the models
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model_params = params['fcNN_params']
		self.params_layers_sizes = self.model_params['params_layers_sizes']
		self.init_hidden_layers() # init self.hidden_layers (a list of nn.Linear with different size specification)
		self.init_activation_func() # init self.actFunc_list  (a list of nn.activation function)
		self.dropout_layer = nn.Dropout(p=self.model_params['dropout_proba'])

		# Second, we will init the components of the training process
		self.training_params = params['training_parameters']
		self.optimizer = optim.Adam(self.parameters(), lr=self.training_params['learning_rate'], weight_decay = self.training_params['l2_regularization'])
		self.start_epoch = 1
		self.best_val_loss = float('inf')
		self.validation_model_fn = os.path.join(model_folder, 'validation.model')
		self.train_model_fn = os.path.join(model_folder, 'train.model')
		self.log_fn = os.path.join(model_folder, 'train.log')
		helper.remove_file(self.log_fn) # remove so that we can write a new one for this particular instance, in case there was a model run before this
		return 

	def init_hidden_layers(self):
		self.mu_bias_init = 0.1
		self.hidden_layers=torch.nn.ModuleDict()
		for layer_index in range(len(self.params_layers_sizes)):
			if layer_index==0:
				self.hidden_layers[str(layer_index)] = nn.Linear(self.model_params['input_dim'], self.params_layers_sizes[layer_index])
				nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
			else:
				self.hidden_layers[str(layer_index)] = nn.Linear(self.params_layers_sizes[layer_index-1],self.params_layers_sizes[layer_index])
				nn.init.constant_(self.hidden_layers[str(layer_index)].bias, self.mu_bias_init)
		return 

	def init_activation_func(self):
		self.actFunc_list = []
		for layer_index in range(len(self.model_params['params_layers_actFuncs'])):
			# set up non-linearity
			if self.model_params['params_layers_actFuncs'][layer_index] == 'relu':
				self.actFunc_list.append(nn.ReLU())
			elif self.model_params['params_layers_actFuncs'][layer_index] == 'tanh':
				self.actFunc_list.append(nn.Tanh())
			elif self.model_params['params_layers_actFuncs'][layer_index] == 'sigmoid':
				self.actFunc_list.append(nn.Sigmoid())
			elif self.model_params['params_layers_actFuncs'][layer_index] == 'elu':
				self.actFunc_list.append(nn.ELU())
			elif self.model_params['params_layers_actFuncs'][layer_index] == 'linear':
				self.actFunc_list.append(nn.Identity())
		return 

	def forward(self, x):
		x = self.dropout_layer(x)  # if dropout_proba is 0 then this is really not doing anything
		for layer_index in range(len(self.params_layers_sizes)):
			x = self.actFunc_list[layer_index](self.hidden_layers[str(layer_index)](x))
			x = self.dropout_layer(x)
		return x

	def get_train_valid_loader(self, data):
		"""
		Function copied from https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb#file-data_loader-py-L14
		and then modified (simplified) by Ha Vu on 01/17/2023
		Utility function for loading and returning train and valid
		multi-process iterators 
		If using CUDA, num_workers should be set to 1 and pin_memory to True.
		Params
		------
		- data: the data object from data_utils.py
		Returns
		-------
		- train_loader: training set iterator.
		- valid_loader: validation set iterator.
		"""
		# load the dataset
		num_train = len(data)
		indices = list(range(num_train))
		split = int(np.floor(self.training_params['validation_set_fract'] * num_train))
		np.random.seed(self.random_seed)
		np.random.shuffle(indices) # this will change indices, inplace
		train_idx, valid_idx = indices[split:], indices[:split]
		train_sampler = SubsetRandomSampler(train_idx)
		valid_sampler = SubsetRandomSampler(valid_idx)
		train_loader = DataLoader(data, batch_size=self.training_params['batch_size'], sampler=train_sampler, num_workers=0)
		valid_loader = DataLoader(data, batch_size=self.training_params['batch_size'], sampler=valid_sampler, num_workers=0)
		# num_workers should be set to 0 otherwise it crashes on my system, Ref: https://stackoverflow.com/questions/60101168/pytorch-runtimeerror-dataloader-worker-pids-15332-exited-unexpectedly
		return (train_loader, valid_loader)

	def validate_model(self, valid_loader, epoch, start, loss_function):
		if not (self.training_params['use_validation_set'] and epoch % self.training_params['validation_freq'] == 0): # if it's not time to validate the model yet, don't do it
			return 
		# now it's time to validate the model
		self.eval()
		with torch.no_grad():
			for batch_index, (x, y) in enumerate(valid_loader, 0):
				x = x.to(self.device) #  batch, bp, alphabet
				y_hat = self.forward(x) # mu, log_var has the same shape: [batch_size (animal), z-dim]
				loss = loss_function(y_hat, y)
				progress_val = "\t\t\t|Val : Update {0}. MSELoss : {1:.3f}, Time: {2:.2f} |".format(epoch, loss.item(), time.time() - start)
				print(progress_val)
				if loss.item() < self.best_val_loss:
					self.best_val_loss =  loss.item()
					self.save_model(self.validation_model_fn)
				return # this meanns that we will onnly look at one batch for validation

	def report_progress(self, epoch, loss, start):
		'''
		During training, this function is called to report on the progress of training and save the model if necessary
		'''
		if epoch % self.training_params['log_training_freq'] == 0:
			progress = "|Train : Update {0}. Loss : {1:.3f}, Time: {2:.2f} |".format(epoch, loss, time.time() - start)
			print(progress)
			logF = open(self.log_fn, "a")
			logF.write(progress+"\n")
		if epoch % self.training_params['save_model_params_freq']==0:
			self.save_model(self.train_model_fn)
		return

	def train_model(self, data):
		if torch.cuda.is_available():
			cudnn.benchmark = True
		self.train() # the train() function may be a function of nn.Module
		train_loader, valid_loader = self.get_train_valid_loader(data)
		start = time.time()
		loss_function = nn.MSELoss(reduction = 'sum') # mean squared error
		for epoch in tqdm.tqdm(range(self.start_epoch ,self.training_params['num_epoch']+1), desc="Training model"):
			for batch_index, (x, y) in enumerate(train_loader, 0): # each time we enumerate data loader, dataloader actually reshuffle at each epoch
				x = torch.tensor(x).to(self.device) #  batch, bp, alphabet
				self.optimizer.zero_grad() # set gradients back to 0 for every trainging iteration. Ref: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
				y_hat = self.forward(x) # mu, log_var has the same shape: [batch_size (animal), z-dim]
				loss = loss_function(y_hat, y)
				loss.backward() # this is required for everything iteration of training
				self.optimizer.step() # also required for every iteration of training
				# now onto validation if it's time to validate
				self.validate_model(valid_loader, epoch, start, loss_function)
				# report progress if it's the time
				self.report_progress(epoch, loss, start)
				if batch_index == (self.training_params['num_batch_per_epoch'] - 1):
					break

	def load_pretrained_model(self, pretrained_model_fn):
		self.pretrained_model = True
		try:
			checkpoint = torch.load(pretrained_model_fn, map_location =  self.device)
			self.load_state_dict(checkpoint['model_state_dict']) # function load_state_dict is part of nn.Module
			print("Initialized FNCC with checkpoint '{}' ".format(pretrained_model_fn))
		except:
			print("Unable to locate FNCC model checkpoint")
			sys.exit(0)
		return 

	def predict_MAF(self, data, batch_size=256):
		"""
		msa_data is the object declared in data_utils.py   
		"""
		dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
		num_samples = len(data)
		prediction_matrix = torch.zeros(num_samples) # (possible mutations, num_samples). Note possible mutations imply num_genomic_positions * 4 (ACTG)
		with torch.no_grad():
			for i, (x,y) in enumerate(tqdm.tqdm(dataloader, 'Looping through input data batches')):
				x = x.to(self.device)
				y_hat = self.forward(x)
				prediction_matrix[i*batch_size:i*batch_size+len(x)] = y_hat.squeeze()
		return prediction_matrix

	def save_model(self, save_fn):
		torch.save({'model_state_dict':self.state_dict(), 'model_parameters':self.model_params, 'training_params': self.training_params}, save_fn)

