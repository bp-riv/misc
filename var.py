import torch
import numpy as np
import matplotlib.pyplot as plt


def historic_cdf (samples, bound):
    '''
    Returns the amount of samples with values
    inferior than the bound, in form of probability [0,1]
    '''
    samples = samples.reshape(-1,)
    satisfy = torch.sum(samples < bound)
    return satisfy / len(samples)
####


def grid_quantile(samples, confidence, n_mesh):
    '''
    
    '''
	samples = samples.reshape(-1,)
	max_sample = torch.max(samples)
	min_sample = torch.min(samples)
	h = (max_sample - min_sample) / n_mesh
    upper_q = max_sample
	lower_q = upper_q - h
	while (condition == False and lower_q > min_sample):
		p_upper = historic_cdf(samples, upper_q)
		p_lower = historic_cdf(samples, lower_q)
		if p_upper >= confidence and p_lower <= confidence:
			condition = True
		else:
			upper_q = lower_q


