import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

def plot_eigs(
        model: torch.nn.Module,
        eigs_r: list[float],
        eigs_im: list[float],
        plot_epoch: Optional[bool],
    ) -> None:
    """Plot eigenvalues over every layer."""
    plt.figure()
    lin = np.linspace(0, 2*np.pi, 1000)
    plt.plot(np.cos(lin), np.sin(lin), linewidth=1, color='k')
    plt.scatter(eigs_r, eigs_im, marker='x', label='Eigs, After Training', linewidth=2, rasterized=True)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    #plt.title(model.dataset)
    plt.savefig(f'plots/eigs_{model.conv_func_name}_{model.dataset}_{plot_epoch}.pdf', dpi=300)
    print(f'SAVING: plots/eigs_{model.conv_func_name}_{model.dataset}_{plot_epoch}.pdf')
    plt.close()

    plt.figure()
    plt.plot(np.cos(lin), np.sin(lin), linewidth=1, color='k')
    plt.hist2d(
        eigs_r, eigs_im, bins=(100,100), range=[(-1.5,1.5),(-1.5,1.5)], cmap='Blues', norm=LogNorm(),
    )
    plt.colorbar()
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    #plt.title(model.dataset)
    plt.savefig(f'plots/eigs_{model.conv_func_name}_{model.dataset}_{plot_epoch}_hist.pdf', dpi=300)
    plt.close()

    mod = 0
    for j in range(len(eigs_r)):
        mod += np.sqrt(eigs_r[j]**2 + eigs_im[j]**2)
    mod = mod / len(eigs_r)
    print('Average Eigenvalue Modulus: ', 1 - mod)
    file = open(f'res/{model.conv_func_name}_{model.dataset}_eig.txt', 'a')
    file.write('Average Eigenvalue Modulus:' + str(1 - mod) + '\n')
    file.close()

    df = pd.DataFrame(columns=['R','I'])
    df['R'] = eigs_r
    df['I'] = eigs_im
    df.to_csv(f'eigs/{model.conv_func_name}_{model.dataset}_{plot_epoch}_eig.csv')


def extract_block_diagonals(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Extracts dxd block diagonals from an nxn square matrix.

    Args:
        x (torch.Tensor): Input tensor of shape (n, n) with block diagonals.
        d (int): Size of each square block.

    Returns:
        torch.Tensor: A tensor of shape (n//d, d, d) containing the block diagonals.

    Raises:
        ValueError: If input tensor is not 2D, not square, or n is not divisible by d.
    """
    # Input validation
    if x.dim() != 2:
        raise ValueError(f"Input tensor must be 2-dimensional, got {x.dim()} dimensions.")

    n, m = x.shape
    if n != m:
        raise ValueError(f"Input tensor must be square, got shape ({n}, {m}).")

    if not isinstance(d, int) or d <= 0:
        raise ValueError(f"Block size 'd' must be a positive integer, got {d}.")

    if n % d != 0:
        raise ValueError(f"Matrix size n={n} is not divisible by block size d={d}.")

    num_blocks = n // d

    # Reshape the matrix to separate the blocks
    # New shape: (num_blocks, d, num_blocks, d)
    try:
        x_reshaped = x.view(num_blocks, d, num_blocks, d)
    except RuntimeError as e:
        raise ValueError(f"Error reshaping tensor: {e}")

    # Permute to bring the blocks into separate dimensions
    # New shape: (num_blocks, num_blocks, d, d)
    x_permuted = x_reshaped.permute(0, 2, 1, 3)

    # Generate indices for diagonal blocks
    block_indices = torch.arange(num_blocks, device=x.device)

    # Extract diagonal blocks
    # Each block is of shape (d, d)
    # The resulting tensor has shape (num_blocks, d, d)
    blocks = x_permuted[block_indices, block_indices]

    return blocks



import math
import torch.nn as nn
import pandas as pd
from torch_geometric.nn import GCNConv, global_mean_pool, ChebConv,global_add_pool
from torch_geometric.utils import to_dense_adj,dense_to_sparse 
from torch.autograd.functional import jacobian

class GCN_Shared_nl(torch.nn.Module):
    def __init__(self, nhid, nlayers, gamma_a, gamma_b, dataset):
        super(GCN_Shared_nl, self).__init__()

        self.nlayers = nlayers

        self.conv = GCNConv(nhid,nhid)
        self.bns = nn.ModuleList()
        self.edge_index = None

        self.n1 = None
        self.n2 = None
        self.fc = nn.Linear(nhid, nhid)
        self.gamma_a = gamma_a
        self.gamma_b = gamma_b

        #self.conv_func_name = 'GCN_nl_ga'+str(self.gamma_a)+'_gb'+str(self.gamma_b)+'_ssm'
        self.conv_func_name = 'Cheb'
        self.plot_epoch = 0
        self.dataset = dataset

        self.nhid = nhid

        params = [self.conv.lin.weight]
        vals = [self.gamma_a, self.gamma_b, 1]
        for i in range(2):
            eivals, eigenvectors = torch.linalg.eig(params[i])
            eigs_r, eigs_im = [], []
            for element in eivals:
                    eigs_r.append(element.real.item())
                    eigs_im.append(element.imag.item())
            df = pd.DataFrame(columns=['R','I'])
            df['R'] = eigs_r
            df['I'] = eigs_im
            df.to_csv(f'eigs/{self.conv_func_name}_{self.dataset}_weight_eig.csv')

            new_eigenvals = torch.ones(len(eigenvectors))*vals[i]

            D = torch.diag(new_eigenvals)  # Diagonal matrix of eigenvalues
            D = torch.tensor(D, dtype=torch.complex64)
            V = eigenvectors  # Matrix of eigenvectors
            new_weight = V @ D @ torch.inverse(V)

            with torch.no_grad():
                params[i].copy_(new_weight.double())

    # def conv_op(self, x):
    #     #print(x.shape)
    #     return x@self.C + torch.relu(self.conv(x, self.edge_index))@self.D

    def forward(self, data):
        x = data.x
        self.edge_index = data.edge_index

        xs = []
        for i in range(self.nlayers):

            x = self.conv_op(x.double())
            if i == 0 and True:
                D = jacobian(self.conv_op_jac_lin, (x.view(-1)))
                #self.conv_func_name = 'GCN_lin_g'+str(self.gamma)
                self.conv_func_name = 'Cheb'
                print('Computing eigenvalues')
                eivals, _ = torch.linalg.eig(D)
                # Compute the modulus of the eigenvalues
                eivals_mod = torch.abs(eivals)

                eigs_r, eigs_im = [], []
                for element in eivals:
                        eigs_r.append(element.real.item())
                        eigs_im.append(element.imag.item())

                print(f'Eigenvalue Modulus Max: {torch.max(eivals_mod)} Mean: {torch.mean(eivals_mod)} Var: {torch.std(eivals_mod)}')

                plot_eigs(
                model=self, eigs_r=eigs_r, eigs_im=eigs_im, plot_epoch=None
                )

            xs.append(x)

        return xs

