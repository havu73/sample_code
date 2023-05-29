import os
import numpy as np
enformer_folder = '/u/home/h/havu73/project-ernst/data/enformer'
gnomad_folder = '/u/home/h/havu73/project-ernst/data/gnomad/v2.1.1_hg19/variants/MAF'
output_folder = '/u/home/h/havu73/project-ernst/source_enformer_expand/base_predict_maf/lm/'
CHROMOSOME_LIST = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22'] # we exclude chromosome Y because it's not available in all cell types
ALPHA_LIST = list(np.linspace(1, 9, 9).astype(int)) + list(np.linspace(10, 100, 10).astype(int))
model_fn = os.path.join(output_folder + 'chr22.model')


rule all:
	input:
		expand(os.path.join(output_folder, 'alpha_{alpha}', 'chr9.pearsonr'), alpha = ALPHA_LIST[:1]),

rule learn_model:
	input:
		os.path.join(enformer_folder, '1000G.MAF_threshold=0.005.22.h5'),
		os.path.join(gnomad_folder, 'chr22_maf.bed.gz'),
	output:
		os.path.join(output_folder, 'alpha_{alpha}', 'chr22.model'),
	shell:
		"""
		python /u/home/h/havu73/project-ernst/source_enformer_expand/base_predict_maf/lm_one_chrom.py --enformer_fn {input[0]} --gnomad_fn {input[1]} --model_fn {output} --ALPHA {wildcards.alpha}
		"""

rule calculate_pearsonR_one_chrom:
	input:
		os.path.join(output_folder, 'alpha_{alpha}', 'chr22.model'),
		os.path.join(enformer_folder, '1000G.MAF_threshold=0.005.{chrom}.h5'),
		os.path.join(gnomad_folder, 'chr{chrom}_maf.bed.gz'),
	output:
		os.path.join(output_folder, 'alpha_{alpha}', 'chr{chrom}.pearsonr'),
	shell:
		'''
		python /u/home/h/havu73/project-ernst/source_enformer_expand/base_predict_maf/lm_one_chrom.py --enformer_fn {input[1]} --gnomad_fn {input[2]} --model_fn {input[0]} --eval_fn {output}
		'''