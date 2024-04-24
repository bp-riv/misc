'''
A very simple Brownian Motion simulation based on scaling walks.
'''

import torch
import math
import matplotlib.pyplot as plt

def simple_bm (mesh_steps : int, time_limit : float):
    '''
    Strategy is to rescale random walks.
    '''
    assert mesh_steps > 0
    assert time_limit > 0
    n_times = int(mesh_steps * time_limit) + 1
    bm = torch.zeros(n_times)
    eta = torch.normal(0., 1., (n_times - 1, ))
    increments = eta / math.sqrt(float(mesh_steps))
    for nth in range(n_times - 1):
#        bm[nth + 1] = bm[nth] + increments[nth]
        # Choosing increments as Bernoulli with mean 0 and variance 1
        inc = torch.bernoulli(torch.tensor(0.5)) * 2. - 1.
        bm[nth + 1] = bm[nth] + inc / math.sqrt(float(mesh_steps))

    return bm
#---


if __name__ == "__main__":
    bm1 = simple_bm (100, 1.2)
    plt.plot(bm1)
    plt.grid()
    plt.show()
