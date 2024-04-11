'''
Simple idea for computing the empirical quantile for a random set
of samples by using sorting.
'''

import torch

def get_quantile (samples, p):
    '''
    samples is assumed to be a torch tensor of at least two
    points, while p a probability
    between 0 and 1.
    '''
    # TO DO: add here the input check
    # Sort samples in descending order
    samples = samples.reshape(-1,).sort(-1, True)[0]
    n_samples = len(samples)
    for nth in range(n_samples - 1):
        q_up = samples[nth]
        q_down = samples[nth + 1]
        # Check if this pairs satisfy the quantile search
        # TO DO:
        # ADD COMPUTATION OF PROBABILTIES AND CONDITION CHECK


    print("Quantile estimated in the interval [{q_down:.2f}, {q_up:.2f}]")
    print(f"P [X <= {q_up:.2f}] = {p_up : .2f}")
    print(f"P [X <= {q_down :.2f}] = {p_down : .2f}")
    print(f" --- desired prob was {p:0.2f} ---")
    return (q_up, q_down)
#---
