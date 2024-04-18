'''
Simple idea for computing the empirical quantile for a random set
of samples by using sorting.
'''

import torch
from math import ceil, floor

# COnverti in numpy, numba und optimieret
# Search abput pydentic base model (class); can vielleicht : float
def get_quantile (og_samples, og_p, verbose = True):
    '''
    samples is assumed to be a torch tensor of at least two
    points, while p a probability
    between 0 and 1.
    TO ADD: is p is higher than 1, interpret as a percentage
    '''
    p = float(og_p)
    # TO CONSTRUCT
    # < 1
    assert(p > 0. and p <= 1.)
    # Sort samples in ascending order
    samples = og_samples.reshape(-1,).sort()[0]
    n_samples = len(samples)
    assert(n_samples >= 2)
    # Convert: probability, e.g. 0.1, into "how many element" to take
    # subtract 1 to convert such a quantity into a valid index
    # TO CONSTRUCT
    idx_low = int(floor(n_samples * p) - 1)
    idx_up = int(ceil(n_samples * p) - 1)

    q_low = samples[idx_low]
    q_up = samples[idx_up]

    # Double check that the two elements approximate the desired quantile
    p_low = torch.sum(samples < q_low) / n_samples
    p_up = torch.sum(samples < q_up) / n_samples

    if verbose:
        print(f"From get_quantile:")
        print(f"\t{p*100:.0f}%-quantile in [{q_low:.2f}, {q_up:.2f}]")
        print(f"\tP [X <= {q_low :.2f}] = {p_low : .2f}")
        print(f"\tP [X <= {q_up:.2f}] = {p_up : .2f}")
        print("-")
    return (q_low, q_up)
#---

def test_get_quantile ():
    n_test1 = 10
    means = torch.randint(20, (n_test1,)) - 10
    for m in means:
        r = torch.normal(m, 1., (100_000,))
        q1, q2 = get_quantile(r, 0.5)
        # The difference between the mean of q1 and q2 must be close to m
        mean_quantile = (q1 + q2) / 2.
        assert (torch.abs(mean_quantile - m) < 0.1)
    # Now a couple of test on the maximum quantile
    r = torch.normal(0., 100., (10_000,))
    _, q_up = get_quantile(r, 1, verbose = False)
    assert (q_up == r.max())
    return 1
#---

def get_var(samples, c_value, verbose = True):
    '''
    Compute the Value At Risk for a set of samples, by using its
    interpretation as quantile: VaR(c) = - QUANTILE(1 - c)
    '''
    p = 1. - c_value
    q_low, q_up = get_quantile(samples, p, verbose = False)
    var_low = - q_up
    var_up = - q_low
    if verbose:
        print(f"{c_value * 100}%-V@R in [{var_low:.2f}, {var_up:.2f}]")
    return (var_low, var_up) 
#---

def test_var ():
    '''
    Testing the previous VaR computation by using a theoretical example
    with known result.
    '''
    n_samples = 10_000
    # Building the two independent random variables
    random_integers_1 = torch.randint(1000, (n_samples,))
    random_integers_2 = torch.randint(1000, (n_samples,))

    samples_1 = torch.ones(n_samples) * 1000.
    samples_2 = torch.ones(n_samples) * 1000.
    for nth in range(n_samples):
        if random_integers_1[nth] > 991:
            samples_1[nth] = -10_000.
        if random_integers_2[nth] > 991:
            samples_2[nth] = -10_000
    # Now, the historical samples are built
    c_value = 0.99 
    var_low, var_high = get_var(samples_1, c_value)
    print("True value is: -1000")
    true_var = -1000.
    err_var = torch.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)

    var_low, var_high = get_var(samples_2, c_value)
    print("True value is: -1000")
    true_var = -1000.
    err_var = torch.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)

    samples_3 = (samples_1 + samples_2) / 2.
    var_low, var_high = get_var(samples_3, c_value)
    print("True value is: 4500")
    true_var = 4500
    err_var = torch.abs((var_low + var_high) / 2. - true_var)
    assert(err_var < 0.1)
    return True
