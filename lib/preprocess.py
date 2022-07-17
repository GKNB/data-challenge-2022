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

def detect_nan_pattern(data_set, data_set_name, dim_along, data_irrelevant):
	nan_idx_list = []
	for var_name in data_set.data_vars:
		if var_name in data_irrelevant:
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
		print("NAN pattern along dimension: {}, is CONSISTENT for all other coords, with {} excluded".format(dim_along, data_irrelevant))
		return {'nan_idx':nan_idx_list[0], 'if_consistent':nan_pattern_consistent} #only need to return one, all others are the same
	else:
		print("NAN pattern along dimension: {}, is NOT CONSISTENT for all other coords, with {} excluded".format(dim_along, data_irrelevant))
		return {'nan_idx':[], 'if_consistent':nan_pattern_consistent} 


def detect_nan_pattern_path(data_set_path, dim_along, data_irrelevant):
	data_set = xr.load_dataset(data_set_path)
	if "name" in data_set.attrs:
		data_set_name = data_set.attrs["name"]
	else:
		data_set_name = data_set_path.split('/')[-1]
	return detect_nan_pattern(data_set, data_set_name, dim_along, data_irrelevant)	

def find_all_nan(dict_of_datasets, dim_along, data_irrelevant):
	print("Searching NAN in DataSets: {}...".format(dict_of_datasets.keys()))
	counter = 0
	for data_set_name in dict_of_datasets:
		nan_detect_result = detect_nan_pattern(dict_of_datasets[data_set_name], data_set_name, dim_along, data_irrelevant)
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

def check_if_nan(dict_of_datasets, dim_along, data_irrelevant):
	print("Checking if there's NAN in datasets:{}".format(dict_of_datasets.keys()))
	if len(find_all_nan(dict_of_datasets, dim_along, data_irrelevant)) == 0:
		return False
	else:
		return True		



def remove_nan(dict_of_datasets, dim_along, data_irrelevant):
	print("Removing NAN indices along dimension:{}".format(dim_along))
	print("WARNING! This will change the oringal DataSets!")
	nan_idx_all = find_all_nan(dict_of_datasets, dim_along, data_irrelevant)
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


def save_standardrized_dict_of_datasets_h5(dict_of_std_datasets, output_folder, file_format, data_format):
	#check if the output_folder exists
	if os.path.exists(output_folder):	
		output_path_format = output_folder+file_format+".h5"
		with h5py.File(output_path_format, 'w') as hf:
			for data_set_name in dict_of_std_datasets:
				for item in dict_of_std_datasets[data_set_name]:
					data_name = (data_format % data_set_name) + "_" + item
					hf.create_dataset(data_name,  data=dict_of_std_datasets[data_set_name][item])
					print("Saving data {} to h5 file {}".format(data_name, output_path_format))
	else:
		print("Output folder: {} doesn't it exist, create one before move on!".format(output_folder) )




def load_standardrized_dict_of_datasets_h5(output_folder, file_format, keyword, exclude_list = ["stddev", "mean"]):
	#check if the output_folder exists
	if os.path.exists(output_folder):	
		output_path_format = output_folder+file_format+".h5"
		with h5py.File(output_path_format, 'r') as hf:
			list_of_keys = list(hf.keys())
			print("Data in file {} are: \n {}".format(output_path_format, list_of_keys) )	
			for data_name in list_of_keys:
				data_name_parsed = data_name.split("_")
				print("Examining data {}".format(data_name))	
				if keyword in data_name_parsed:
					excluded = True
					for exclude_term in exclude_list:
						if exclude_term in data_name_parsed:
							excluded = False
					if excluded == True:
						print("Loading data {} from file {}".format(data_name, output_path_format))
						return np.array(hf[data_name][:])	
			print("Data containing keyword: {}, with terms {} excluded, not found in file {}".format(keyword, exclude_list, output_path_format))	
	else:
		print("Output folder: {} doesn't it exist, create one before move on!".format(output_folder) )


def get_data_xy_from_h5(output_folder, file_format, xy_keyword_dict, exclude_list = ["stddev", "mean"]):
	data_x = load_standardrized_dict_of_datasets_h5(output_folder, file_format, xy_keyword_dict["x"], exclude_list)
	data_y = load_standardrized_dict_of_datasets_h5(output_folder, file_format, xy_keyword_dict["y"],exclude_list)
	print("Got data_x with shape:{}, data_y with shape:{}".format(data_x.shape, data_y.shape))
	return {"data_x":data_x, "data_y":data_y}






def check_IO_standardrized_dict_of_datasets_h5(dict_of_std_datasets, output_folder, file_format, data_format):
	#check if the output_folder exists
	if os.path.exists(output_folder):
		IO_safe = True		
		output_path_format = output_folder+file_format+".h5"
		with h5py.File(output_path_format, 'r') as hf:
			for data_set_name in dict_of_std_datasets:
				for item in dict_of_std_datasets[data_set_name]:
					data_name = (data_format % data_set_name) + "_" + item
					original_data=dict_of_std_datasets[data_set_name][item]
					reopen_data = hf[data_name][:]
					if not np.array_equal(original_data,reopen_data):
						IO_safe = False
						print("IO check of data {} to h5 file {} FAILED".format(data_name, output_path_format))
			if IO_safe == True:
				print("IO check of data set:{} to h5 file {} PASSED".format(dict_of_std_datasets.keys(), output_path_format))
			return IO_safe
	else:
		print("Output folder: {} doesn't it exist, create one before move on!".format(output_folder) )




def stack_data_set_datas(data_dict, stacking_data_list, item_name):
	stacking_dim_idx = 0
	stacking_count = 0
	for data_name in stacking_data_list:
		print("\t\tStacking data {}, {} with mean:{}, std:{}".format(data_name, item_name, data_dict[data_name].mean(), data_dict[data_name].std()))
		if stacking_count == 0:
			stacked_data = np.array(data_dict[data_name])
		elif stacking_count > 0:
			 stacked_data = np.stack((stacked_data, data_dict[data_name]), axis=stacking_dim_idx)
		else:
			print("Error! Invalid stacking_count number!")
		stacking_count = stacking_count + 1
	axis_idx_list = list(range(len(stacked_data.shape)))
	del axis_idx_list[0]
	axis_idx_list.append(0)
	#move the first idx last
	stacked_data_out =stacked_data.transpose(tuple(axis_idx_list))
	print("\tStacked {} data in data_dict: {}".format(stacking_count, stacking_data_list))	

	return stacked_data_out
	
def check_data_shape_valid(data):
	if len(data.shape) == 3:
		print("shape check: PASSED. This program expect data with 3 dimensions:[time, x, y] for 2D training.")
		return True
	else:
		print("Error! This program expect data with 3 dimensions:[time, x, y] for 2D.")
		return False

def check_standardrize_dataset_GAN(data_set, target_data_list, std_data_stacked, tolerance):
	print("\tReconstruct stacked unstandardrized data!")
	print("\t data_set_recovered = std_data[{}] * stddev[{}] + mean[{}]".format(std_data_stacked["std"].shape, std_data_stacked["stddev"].shape, std_data_stacked["mean"].shape))
	data_set_recovered = std_data_stacked["std"] * std_data_stacked["stddev"] + std_data_stacked["mean"]
	data_set_raw = dict()
	for data_name in target_data_list:		
		data_set_raw[data_name] = data_set[data_name].to_numpy()	
	raw_stacked = stack_data_set_datas(data_set_raw, target_data_list, "raw")		
	num_error = np.max(raw_stacked - np.array(data_set_recovered) )
	print("\tNumerical error of standardrization is: {}".format(num_error))
	if num_error < tolerance:
		return True
	else:
		return False




def standardrize_dataset_GAN(data_set, target_data_list, stat_dim):
	#get statistics
	mean_dict = dict()	
	stddev_dict = dict()
	std_dict = dict()
	for data_name in target_data_list:
		print("\tStandardrizing data:%s" % data_name)
		assert(check_data_shape_valid(data_set[data_name]))
		#confirmed the mean and std is over all dimension, not only stat dimension	
		mean_dict[data_name] = data_set[data_name].mean().to_numpy()
		stddev_dict[data_name] = data_set[data_name].std().to_numpy()

		print("\tmean_shape:{}, stddev_shape:{}, std_shape:{}".format(mean_dict[data_name].shape, stddev_dict[data_name].shape, data_set[data_name].to_numpy().shape))
		std_dict[data_name] = (data_set[data_name].to_numpy() - mean_dict[data_name] )\
								 / stddev_dict[data_name]
	print("\tStacking standardrized data: {}".format(target_data_list))	
	std_stacked = stack_data_set_datas(std_dict, target_data_list, "stdrzd")
	mean_stacked = stack_data_set_datas(mean_dict, target_data_list, "mean")
	stddev_stacked = stack_data_set_datas(stddev_dict, target_data_list, "std")
	print("\tstacked: mean_shape:{}, stddev_shape:{}, std_shape:{}".format(mean_stacked.shape, stddev_stacked.shape, std_stacked.shape) )
	axis_idx_list = list(range(len(std_stacked.shape)))
	del axis_idx_list[-1]
	stat_axis_tuple = tuple(axis_idx_list)
	print("\tstacked std stat(over all other dimensions {}): mean:{}, stddev:{}".format(stat_axis_tuple, std_stacked.mean(axis=stat_axis_tuple), std_stacked.std(axis=stat_axis_tuple)) )
	return {"std":std_stacked, "mean":mean_stacked, "stddev":stddev_stacked}



def standardrize_dict_of_datasets_GAN(dict_of_datasets, target_dataset_list, target_data_list, stat_dim):
	dict_of_std_datasets = dict()
	for data_set_name in target_dataset_list:
		print("Standardrizing data set: {}".format(data_set_name))
		dict_of_std_datasets[data_set_name] = standardrize_dataset_GAN(dict_of_datasets[data_set_name], target_data_list, stat_dim)
	return deepcopy(dict_of_std_datasets)


def check_standardrize_dict_of_datasets_GAN(dict_of_datasets, target_dataset_list, target_data_list, dict_of_std_datasets, tolerance):
	std_check_passed = True
	for data_set_name in target_dataset_list:
		print("Checking standardrized data set: {}".format(data_set_name))
		if not check_standardrize_dataset_GAN(dict_of_datasets[data_set_name], target_data_list, dict_of_std_datasets[data_set_name], tolerance):
			std_check_passed = False
			print("Standardrization check FAILED!")
	if std_check_passed == True:
		print("Standardrization check for standardrized data set: {}, PASSED".format(data_set_name))
	return std_check_passed




def data_preprocess(list_of_data_set_path, parameters):
	dim_along = parameters["nan_dim_along"]
	data_irrelevant = parameters["nan_data_irrelevant"]
	output_folder = parameters["output_folder"] 
	file_format =  parameters["file_format"]
	dict_of_datasets = create_data_set_dict(list_of_data_set_path)
	remove_nan(dict_of_datasets, dim_along, data_irrelevant)
	#check if nan have been removed
	if check_if_nan(dict_of_datasets, dim_along, data_irrelevant):
		print("NAN not removed! Check the data more carefully!")
		return
	else:
		print("NAN all get removed! Saving preprocessed data!")
		save_dict_of_datasets(dict_of_datasets, output_folder, file_format)
		check_NAN_IO = check_IO_dict_of_datasets(dict_of_datasets, output_folder, file_format)

	#standardrize the data and save it
	std_file_format = parameters["std_file_format"]
	data_format = parameters["std_data_format"]
	target_data_list = parameters["std_data_list"]
	target_dataset_list = parameters["std_dataset_list"]
	stat_dim = parameters["stat_dim"]
	tolerance = parameters["num_error_tolerance"]
	dict_of_std_datasets = standardrize_dict_of_datasets_GAN(dict_of_datasets, target_dataset_list, target_data_list, stat_dim)
	check_STD_GAN = check_standardrize_dict_of_datasets_GAN(dict_of_datasets, target_dataset_list, target_data_list,dict_of_std_datasets, tolerance)	
	save_standardrized_dict_of_datasets_h5(dict_of_std_datasets, output_folder, std_file_format, data_format)
	check_STD_GAN_IO = check_IO_standardrized_dict_of_datasets_h5(dict_of_std_datasets, output_folder, std_file_format, data_format)
	if check_STD_GAN_IO and check_STD_GAN and check_NAN_IO:
		print("Successfully preprocessed all the data!")
	else:
		print("Failed to preprocessed all the data! Check the log for errors!")	

