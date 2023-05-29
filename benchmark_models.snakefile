import os
default_model_config = 'model_config.json'
all_model_output = 'output' # usually a full-path, but for demonstration purposes (portable to other peoples' machine), I keep it as relative path
enformer_folder = 'data/enformer'
gnomad_folder = 'data/gnomad'
input_dim = 5313
num_layers_list = range(1,3) 
CHROM_LIST = ['9'] # for demo purpose, we will only train on chrom 22 and test on chrom 9

rule all:
	input:
		expand(os.path.join(all_model_output, 'nLayer_{num_layer}', 'chr{chrom}.pearsonr'), num_layer = num_layers_list, chrom = CHROM_LIST),

def create_layer_list(wildcards):
	result = []
	current_dim = input_dim # global variable
	num_layer = int(wildcards.num_layer) # any wildcards are str so needs conversions
	for layer in range(num_layer-1): # we need the -1 because we already included 1 as the last layer dimension
		current_dim = int(current_dim/2)
		if current_dim == 1: # the dimension has declined enough to 1, we can stop the layers now
			break
		result.append(current_dim)
	result.append(1)
	return result

rule create_model_config:
	input:
		default_model_config,
	output:
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'model_config.json'),
	params:
		layer_sizes = create_layer_list, 
	shell:
		"""
		python generate_model_config.py --default_fn {default_model_config} --output_fn {output} --params_layers_sizes {params.layer_sizes}
		"""

rule learn_model:
	'''
	learn model on just chrom 22 for now
	'''
	input:
		os.path.join(enformer_folder, '1000G.MAF_threshold=0.005.22.h5'), # for demo purposes, we will only train models on chrom22
		os.path.join(gnomad_folder, 'chr22_maf.bed.gz'),
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'model_config.json'), # from rule create_model_config
	output:
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'train.model'), # model that got saved along training process
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'validation.model'), # model that performed best on validation data		
	params:
		model_folder = os.path.join(all_model_output, 'nLayer_{num_layer}'),
	shell:
		"""
		python apply_FCNN.py --enformer_fn {input[0]} --gnomad_fn {input[1]} --model_folder {params.model_folder} --model_params_fn {input[2]}
		"""

rule calculate_pearsonR_one_chrom:
	input:
		os.path.join(enformer_folder, '1000G.MAF_threshold=0.005.{chrom}.h5'),
		os.path.join(gnomad_folder, 'chr{chrom}_maf.bed.gz'),
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'model_config.json'), # model architecture, from rule create_model_config
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'validation.model'), # model that performed best on validation data, from rule learn_model	
	output:
		os.path.join(all_model_output, 'nLayer_{num_layer}', 'chr{chrom}.pearsonr'),
	params:
		model_folder = os.path.join(all_model_output, 'nLayer_{num_layer}'),
	shell:
		'''
		python apply_FCNN.py --enformer_fn {input[0]} --gnomad_fn {input[1]} --model_folder {params.model_folder} --eval_fn {output} --model_params_fn {input[2]}
		'''		
