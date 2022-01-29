import os
import sys
# print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import numpy as np
import torch
from hamiltorch_sourcecode.samplers import *
from hamiltorch_sourcecode.util import *
import matplotlib.pyplot as plt
import time

import torch.nn as nn
import torch.nn.functional as F

from utils import *


# device
set_random_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_prob(omega):
    mean = torch.tensor([0.,0.,0.])
    stddev = torch.tensor([.5,1.,2.]) 
    return torch.distributions.MultivariateNormal(mean, torch.diag(stddev**2)).log_prob(omega).sum()

def funnel_ll(w):
    v_dist = torch.distributions.Normal(0,3)
    ll = v_dist.log_prob(w[0])
    x_dist = torch.distributions.Normal(0,torch.exp(-w[0])**0.5)
    ll += x_dist.log_prob(w[1:]).sum()
    return ll

if __name__ == "__main__":
    # is_file_load
    is_file_load = False
    
    # Dimension
    D = 10

    # Explicit RMHMC with SOFTABS
    set_random_seed(123)
    params_init = torch.vstack((torch.ones(D + 1), torch.ones(D + 1)))

    # DEBUG
    print(params_init.shape)
    print(params_init)

    params_init[:, 0] = 0.
    step_size = 0.14 
    num_samples = 100 # For results in plot num_samples = 1000
    L = 25
    omega=10
    softabs_const=10**6
    jitter=0.001

    # explicit RMHMC
    if not is_file_load:
        # process time
        start = time.time()

        # DEBUG
        print(params_init.shape)
        print(params_init)

        params_e_rmhmc = sample(log_prob_func=funnel_ll, params_init=params_init, num_samples=num_samples,
                                    sampler=Sampler.RMHMC, integrator=Integrator.EXPLICIT,
                                    metric=Metric.SOFTABS, jitter=jitter,
                                    num_steps_per_sample=L, step_size=step_size, explicit_binding_const=omega, 
                                    softabs_const=softabs_const, debug=True)

        # end time
        end = time.time()

        # print
        print("process time :{:.2f}".format(end-start), "seconds")

        # Covert to numpy arrays for plotting
        coords_e_rmhmc = torch.cat(params_e_rmhmc).reshape(len(params_e_rmhmc),-1).numpy()
        # save file
        np.save("./result_RMHMC.npy", coords_e_rmhmc)
    
    # DEBUG
    else:
        coords_e_rmhmc = np.load('./result_RMHMC.npy')

    # Plotting
    plot_samples(coords_e_rmhmc)