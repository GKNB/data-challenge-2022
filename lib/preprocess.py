import numpy as np
import pandas as pd
import xarray as xr
import h5py
from copy import deepcopy
import os
import matplotlib.pyplot as plt

#check if there is nan in a data set
def detect_nan(data_set, data_set_name):
	print("detect_nan for DataSet: {}".format(data_set_name))	
	for var_name in data_set.data_vars:
		print('\t {}: {}'.format(var_name, np.isnan(data_set[var_name]).any().to_numpy()) )

def detect_nan_path(data_set_path):
	data_set = xr.load_dataset(data_set_path)
	if "name" in data_set.attrs:
		data_set_name = data_set.attrs["name"]
	else:
		data_set_name = data_set_path.split('/')[-1]
	detect_nan(data_set, data_set_name)






# Check the pattern where nan shows up
# The pattern should be that, for a given time, we see nan on all sites for all variables dim_along

def detect_nan_pattern(data_set, data_set_name, dim_along, dim_irrelevant):
	nan_idx_list = []
	for var_name in data_set.data_vars:
		if var_name in dim_irrelevant:
			continue
		print("Checking nan pattern of variable:", var_name, " for DataSet: %s" % data_set_name)
		mask_nan = np.isnan(data_set[var_name])
		nan_idx_tuple = np.where(mask_nan == True)
		dim_along_idx = data_set[var_name].get_axis_num(dim_along)
		nan_dim_along_idx = np.unique(nan_idx_tuple[dim_along_idx])
		print('Total number of NAN: {}, along {} {} indicies'.format(nan_idx_tuple[dim_along_idx].shape, nan_dim_along_idx.shape, dim_along) )
		nan_idx_list.append(nan_dim_along_idx)
		#    print(nan_t_idx, '\n')
		#check if the pattern persist in the neighboring elements
	nan_pattern_consistent = True
	for i in range(len(nan_idx_list) -1):
		if not np.array_equal(nan_idx_list[i], nan_idx_list[i+1]):
			nan_pattern_consistent = False
	if nan_pattern_consistent == True:
		print("NAN pattern along dimension: {}, is CONSISTENT for all other coords, with {} excluded".format(dim_along, dim_irrelevant))
		return {'nan_idx':nan_idx_list[0], 'if_consistent':nan_pattern_consistent} #only need to return one, all others are the same
	else:
		print("NAN pattern along dimension: {}, is NOT CONSISTENT for all other coords, with {} excluded".format(dim_along, dim_irrelevant))
		return {'nan_idx':[], 'if_consistent':nan_pattern_consistent} 


def detect_nan_pattern_path(data_set_path, dim_along, dim_irrelevant):
	data_set = xr.load_dataset(data_set_path)
	if "name" in data_set.attrs:
		data_set_name = data_set.attrs["name"]
	else:
		data_set_name = data_set_path.split('/')[-1]
	return detect_nan_pattern(data_set, data_set_name, dim_along, dim_irrelevant)	

def find_all_nan(dict_of_datasets, dim_along, dim_irrelevant):
	print("Searching NAN in DataSets: {}...".format(dict_of_datasets.keys()))
	counter = 0
	for data_set_name in dict_of_datasets:
		nan_detect_result = detect_nan_pattern(dict_of_datasets[data_set_name], data_set_name, dim_along, dim_irrelevant)
		data_set_nan_idx = nan_detect_result['nan_idx']
		if_nan_consistent = nan_detect_result['if_consistent']
		if if_nan_consistent == False:
			print("Error! NAN pattern not consistent in DataSet: {}".format(data_set_name))
			return []
		else:
			if counter == 0:
				nan_idx_all = np.array(data_set_nan_idx)
			else:
				nan_idx_all = np.union1d(nan_idx_all, data_set_nan_idx)
			counter = counter + 1
	return nan_idx_all	

def check_if_nan(dict_of_datasets, dim_along, dim_irrelevant):
	print("Checking if there's NAN in datasets:{}".format(dict_of_datasets.keys()))
	if len(find_all_nan(dict_of_datasets, dim_along, dim_irrelevant)) == 0:
		return False
	else:
		return True		



def remove_nan(dict_of_datasets, dim_along, dim_irrelevant):
	print("Removing NAN indices along dimension:{}".format(dim_along))
	print("WARNING! This will change the oringal DataSets!")
	nan_idx_all = find_all_nan(dict_of_datasets, dim_along, dim_irrelevant)
	print("nan indices to be removed are: {}".format(nan_idx_all)) 
	for data_set_name in dict_of_datasets:
		exec('dict_of_datasets[data_set_name] = dict_of_datasets[data_set_name].drop_isel(%s =nan_idx_all)'	% dim_along)
	

def create_data_set_dict(list_of_data_set_path):
	print("Creating DataSets by loading files:{}".format(list_of_data_set_path))
	dict_of_datasets = dict()
	for data_set_path in list_of_data_set_path:
		data_set_tmp = xr.load_dataset(data_set_path)
		if "name" in data_set_tmp.attrs:
			data_set_name = data_set_tmp.attrs["name"]
		else:
			data_set_name = data_set_path.split('/')[-1].strip(".nc")
			data_set_tmp.attrs["name"] = data_set_name
		if data_set_name in dict_of_datasets:
			data_set_name = data_set_name + '_new'
			data_set_tmp.attrs["name"] = data_set_name
			print("Same DataSet name found! Create new name:{}".format(data_set_name_new))
		dict_of_datasets[data_set_name] = data_set_tmp
	return deepcopy(dict_of_datasets)


def check_IO_dict_of_datasets(dict_of_datasets, output_folder, file_format):
	output_path_format = output_folder+file_format
	IO_safe = True
	for data_set_name in dict_of_datasets:
		reopen_name = (output_path_format % data_set_name) +".nc"
		print("Loading again DataSet: {} from {}".format(data_set_name, reopen_name))
		dataset_reopen_tmp = xr.load_dataset(reopen_name)
		if not dict_of_datasets[data_set_name].equals(dataset_reopen_tmp):
			IO_safe = False
			print("DataSet:{} is NOT IO_safe.".format(data_set_name))
	if IO_safe == True:
		print("DataSets:{} are all IO_safe.".format(dict_of_datasets.keys()) )
	
	return IO_safe 

		
def save_dict_of_datasets(dict_of_datasets, output_folder, file_format):
	#check if the output_folder exists
	if os.path.exists(output_folder):	
		output_path_format = output_folder+file_format
		for data_set_name in dict_of_datasets:
			save_name = (output_path_format % data_set_name) +".nc"
			print("Saving DataSet: {} to {}".format(data_set_name, save_name))
			dict_of_datasets[data_set_name].to_netcdf(save_name)
	else:
		print("Output folder: {} doesn't it exist, create one before move on!".format(output_folder) )


def data_preprocess(list_of_data_set_path, parameters):
	dim_along = parameters["nan_dim_along"]
	dim_irrelevant = parameters["nan_dim_irrelevant"]
	output_folder = parameters["output_folder"] 
	file_format =  parameters["file_format"]
	dict_of_datasets = create_data_set_dict(list_of_data_set_path)
	remove_nan(dict_of_datasets, dim_along, dim_irrelevant)
	#check if nan have been removed
	if check_if_nan(dict_of_datasets, dim_along, dim_irrelevant):
		print("NAN not removed! Check the data more carefully!")
	else:
		print("NAN all get removed! Saving preprocessed data!")
		save_dict_of_datasets(dict_of_datasets, output_folder, file_format)
		check_IO_dict_of_datasets(dict_of_datasets, output_folder, file_format)
		return
