from csbdeep.utils import _raise
import random
import os

def substract_dict(dict1,dict2):
	"""
		substract two dictionary

	"""
	subtracted_dict = {k:((dict1[k]-dict2[k]) if k in dict2 else dict1[k]) for k in dict1}
	for k in dict2:
		if k in subtracted_dict:
			continue
		else:
			subtracted_dict[k] = dict2[k]
	
	return subtracted_dict

def add_dict(dict1,dict2):
	"""
		add two dictionary

	"""
	added_dict = {k:((dict1[k]+dict2[k]) if k in dict2 else dict1[k]) for k in dict1}
	for k in dict2:
		if k in added_dict:
			continue
		else:
			added_dict[k] = dict2[k]
	
	return added_dict

def getFilePathList(path, filetype):
    pathList = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(filetype):
                pathList.append(os.path.join(root, file))
    return pathList

def staircase_exponential_decay(n):
    '''
    Returns a scheduler function to drop the learning rate by half
    every `n` epochs.
    '''
    return lambda epoch, lr: lr / 2 if epoch != 0 and epoch % n == 0 else lr