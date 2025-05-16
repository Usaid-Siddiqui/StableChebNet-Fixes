from torch_geometric.nn.conv import MessagePassing, ChebConv
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian
from torch.nn import Module
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jacobian
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn.utils.parametrize import register_parametrization
import os

class AntiSymmetric(Module):
    r"""
    Anti-Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} - \mathbf{W}^T`
    """
    def __init__(self, dissipative_force=0.0):
        super().__init__()
        self.g = dissipative_force

    def forward(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T - self.g * torch.eye(W.shape[0]).to(W.device)

    def right_inverse(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1)


class Euler_ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        step_size: float = 0.1,
        dissipation_force: float = 0.0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = 'sym'
        self.e = step_size
        self.g = dissipation_force
        self.lins = torch.nn.ModuleList()
        for _ in range(K):
            self.lins.append(Linear(in_channels, out_channels, bias=False, weight_initializer='glorot'))
            register_parametrization(self.lins[-1],'weight', AntiSymmetric(dissipative_force=self.g))

        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins[1:]:
            lin.reset_parameters()
        zeros(self.bias)


    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
        eig: bool = False
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        self.edge_index, self.norm = edge_index, norm
        self.n1, self.n2 = x.shape[0], x.shape[1]

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        out = x + self.e * out

        if eig:
          eigs_r = []
          eigs_im = []
          J = jacobian(self.conv_jac, (x.view(-1)), create_graph=True)

          print('Computing Eigenvalues')

          eigs = torch.linalg.eigvals(J).detach().cpu().numpy().tolist()
          for element in eigs:
                  eigs_r.append(element.real)
                  eigs_im.append(element.imag)

          # make two subplots: the first is the distribution of the eigenvalues and the second is the plot of the eigenvalues in the complex plane
          fig, axs = plt.subplots(1,2)
          axs[0].hist(eigs_r, bins=100, label='Real', alpha=0.5)
          axs[0].hist(eigs_im, bins=100, label='Imaginary', alpha=0.5)
          axs[0].set_xlabel('Eigenvalue')
          axs[0].set_ylabel('Frequency')
          axs[0].set_title(f'Distribution of Eigenvalues')
          axs[0].legend()
          lin = np.linspace(0, 2*np.pi, 1000)
          axs[1].plot(np.cos(lin), np.sin(lin), linewidth=1, color='k')
          axs[1].scatter(eigs_r, eigs_im, marker='x', label='Eigs, After Training', linewidth=2, rasterized=True)
          axs[1].set_xlabel('Real')
          axs[1].set_ylabel('Imaginary')
          axs[1].set_title(f'Eigenvalues')
          plt.show()
        return out

    def conv_jac(self, x: Tensor) -> Tensor:
        x = x.reshape(self.n1, self.n2)
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            Tx_1 = self.propagate(self.edge_index, x=x, norm=self.norm)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            Tx_2 = self.propagate(self.edge_index, x=Tx_1, norm=self.norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        out = x + self.e * out
        return out.view(-1)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')

