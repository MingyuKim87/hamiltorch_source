import torch
import hamiltorch
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F

def plot_samples(coords, file_name="./result_coords.png"):
    xlim = [-4,4]
    ylim = [0,7]#[-2,9]
    text_x = -1.5
    text_y = 8
    font_size_text = 20
    fs=17
    vxx = torch.linspace(xlim[0],xlim[1],300)
    p = torch.distributions.Normal(0,3)
    v_pdf = torch.exp(p.log_prob(vxx))

    plt.scatter(coords[:,1], coords[:,0], s=5, alpha=0.3,rasterized=True, color='C1')
    plt.legend(loc=0, fontsize=fs)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(xlim)
    plt.tick_params(axis='both', labelsize=fs)
    plt.xlabel(r'$x_l', fontsize=font_size_text)

    plt.savefig(file_name, dpi=300)
    plt.clf()


