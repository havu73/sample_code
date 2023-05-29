import os
import numpy as np
import pandas as pd
import h5py

def create_sample_data(enformer_fn, gnomad_fn, output_folder, chrom):
N_ROWS = 100000
gno_df = pd.read_csv(gnomad_fn, header = None, sep = '\t', index_col = None)
gno_df = gno_df.loc[:N_ROWS]
save_gno_fn = os.path.join(output_folder, 'chr{}_maf.bed.gz'.format(chrom))
gno_df.to_csv(save_gno_fn, header = False, index = False, sep = '\t', compression = 'gzip')
enf_f = h5py.File(enformer_fn, 'r')
save_enf_fn = os.path.join(output_folder, '1000G.MAF_threshold=0.005.{}.h5'.format(chrom)) 
file = h5py.File(save_enf_fn, "w")
for feature in ['SAD', 'alt', 'chr', 'pos', 'ref', 'snp']:
	df = enf_f[feature][:N_ROWS]
	file.create_dataset(feature, data = df)
	file.create_dataset('target_ids', data = enf_f['target_ids'])
	file.create_dataset('target_labels', data = enf_f['target_labels'])
	file.close()
	return 

out_maf_fn = '/u/home/h/havu73/project-ernst/source_enformer_expand/sample_code/data/chr9_maf.bed.gz'
out_enf_fn = '/u/home/h/havu73/project-ernst/source_enformer_expand/sample_code/data/1000G.MAF_threshold=0.005.9.h5'