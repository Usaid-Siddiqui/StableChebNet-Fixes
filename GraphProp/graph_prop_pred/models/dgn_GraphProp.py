# This file contains the implementation of the DGN_GraphProp model.
import torch
import torch.nn as nn
from torch.nn import Module, Linear, ModuleList, Sequential, LeakyReLU
from torch_geometric.data import Data
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from typing import Optional
from collections import OrderedDict
from torch import tanh
from torch_geometric.nn import ChebConv, BatchNorm#, BatchNorm1d
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.functional import jacobian
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian
from torch.nn.utils.parametrize import register_parametrization
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


class DiagLinear(Module):
    def __init__(self, in_channels, g = None):
        super().__init__()
        if g is None:
            # We learn W
            self.W = torch.nn.Parameter(torch.ones(in_channels,dtype=torch.float))
        else:
            # W is fixed
            self.W = torch.nn.Parameter((1-g) * torch.ones(in_channels,dtype=torch.float), requires_grad=False)
    
    def reset_parameters(self):
        pass

    def forward(self, x):
        return x * self.W



class NonDissipv2_ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        K: int,
        dissipation_term: float = 0.0, # it pushes the eigenvalues toward 0
        eigenval_scaler: float = 1.0, # it scales the eigenvalues
        learn_additional_terms: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.normalization = 'sym'
        self.K = K
        self.lins = torch.nn.ModuleList()
        self.lins.append(DiagLinear(in_channels, g=None if learn_additional_terms else dissipation_term)) # this add dissipation to the model
        for i in range(1, K):
            self.lins.append(
                Linear(in_channels, in_channels, bias=False,weight_initializer='glorot')
            )
            register_parametrization(self.lins[i], 'weight', AntiSymmetric())
        
        if learn_additional_terms:
            self.eigenval_scaler = torch.nn.ModuleList([
                torch.nn.Sequential(
                    Linear(in_channels, in_channels), 
                    torch.nn.ELU(), 
                    Linear(in_channels,1),
                    torch.nn.Sigmoid()
                ) for _ in range(K-1)
            ])
        else:
            self.eigenval_scaler = [eigenval_scaler for _ in range(K-1)]
        self.learn_additional_terms = learn_additional_terms

        if bias:
            self.bias = Parameter(Tensor(in_channels))
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

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None, 
                batch: OptTensor = None, lambda_max: OptTensor = None, eig: bool = False) -> Tensor:

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
            eigenval_scaler = self.eigenval_scaler[0](Tx_1) if self.learn_additional_terms else self.eigenval_scaler[0] # eigenval scaler has k-1 elements
            out = out + eigenval_scaler * self.lins[1](Tx_1)
            
        for lin, eigenval_scaler in zip(self.lins[2:], self.eigenval_scaler[1:]):
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            eigenval_scaler = eigenval_scaler(Tx_2) if self.learn_additional_terms else eigenval_scaler
            out = out + eigenval_scaler * lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

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
            eigenval_scaler = self.eigenval_scaler(Tx_1) if self.learn_additional_terms else self.eigenval_scaler
            out = out + eigenval_scaler * self.lins[1](Tx_1)
            
        for lin in self.lins[2:]:
            Tx_2 = self.propagate(self.edge_index, x=Tx_1, norm=self.norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            eigenval_scaler = self.eigenval_scaler(Tx_2) if self.learn_additional_terms else self.eigenval_scaler
            out = out + eigenval_scaler * lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out.view(-1)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j



class Euler_ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        step_size: float = 0.5,
        dissipation_force: float = 0.01,
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


class NonDissip_ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = 'sym'
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K-1)
        ])
        for i in range(K-1):
            register_parametrization(self.lins[i],'weight', AntiSymmetric())

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
            batch=batch,)

        self.edge_index, self.norm = edge_index, norm
        self.n1, self.n2 = x.shape[0], x.shape[1]

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = Tx_0

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) >= 1:
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            out = out + self.lins[0](Tx_1)


        for lin in self.lins[1:]:
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

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
        out = Tx_0

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) >= 1:
            Tx_1 = self.propagate(self.edge_index, x=x, norm=self.norm)
            out = out + self.lins[0](Tx_1)

        for lin in self.lins[1:]:
            Tx_2 = self.propagate(self.edge_index, x=Tx_1, norm=self.norm)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias
        return out.view(-1)

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')
    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DGN_GraphProp(Module):
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 K: int, ## Chebyshev order
                 epsilon: float,
                 dissipation_force: float,
                 hidden_dim: Optional[int] = None,
                 num_layers: int = 1,
                 node_level_task: bool = False,
                 conv_layer: str = 'GCNConv',
                 
                 alpha: Optional[float] = None) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha
        self.bn = nn.BatchNorm1d(hidden_dim)
        

        inp = self.input_dim
        self.emb = None
        if self.hidden_dim is not None:
            self.emb = Linear(self.input_dim, self.hidden_dim)
            inp = self.hidden_dim

        if conv_layer != 'ChebConvDis':
            if conv_layer!='Euler':
                if conv_layer!='Euler2':
                    if conv_layer!='NonDisCheb':
                 
                        self.conv_layer = getattr(pyg_nn, conv_layer)
        elif conv_layer == 'ChebConvDis':  
            self.conv_layer = 'ChebConvDis'

        elif conv_layer == 'NonDisCheb':  
            self.conv_layer = 'NonDisCheb'

        elif conv_layer == 'Euler':
            self.conv_layer = 'Euler'

        elif conv_layer == 'Euler2':
            self.conv_layer = 'Euler2'
        else:
            print('Invalid Convolution Layer')
    
        self.conv = ModuleList()
        for _ in range(num_layers):

            if conv_layer == 'GINConv':
                mlp = Linear(inp, inp)
                self.conv.append(self.conv_layer(nn=mlp,
                                                 train_eps = True))
            elif conv_layer == 'GCN2Conv':
                self.conv.append(self.conv_layer(channels = inp,
                                                 alpha = self.alpha))
            elif conv_layer == 'ChebConv':
              self.conv.append(self.conv_layer(in_channels = inp,
                                                 out_channels = inp,K=K))
            elif conv_layer == 'ChebConvDis':
              self.conv.append(NonDissip_ChebConv(in_channels = inp,
                                                 out_channels = inp,K=K))
            elif conv_layer == 'NonDisCheb':
              self.conv.append(NonDissipv2_ChebConv(in_channels = inp,K=K)) 
            elif conv_layer == 'Euler2':
                self.conv.append(Euler_ChebConv(in_channels = inp,
                                                     out_channels = inp,K=K,step_size=epsilon,dissipation_force=dissipation_force))
            elif conv_layer == 'Euler':
              self.conv.append(Euler_ChebConv(in_channels = inp,
                                                 out_channels = inp,K=K,step_size=epsilon,dissipation_force=dissipation_force)) 
              
            #   print("NonDissipito")
            # else:
            #     self.conv.append(self.conv_layer(in_channels = inp,
            #                                      out_channels = inp))

        self.node_level_task = node_level_task 
        if self.node_level_task:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp, inp // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear(inp // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))
        else:
            self.readout = Sequential(OrderedDict([
                ('L1', Linear(inp * 3, (inp * 3) // 2)),
                ('LeakyReLU1', LeakyReLU()),
                ('L2', Linear((inp * 3) // 2, self.output_dim)),
                ('LeakyReLU2', LeakyReLU())
            ]))

        


    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x= x.to(device)
        # print(x.device)

        x = self.emb(x) if self.emb else x

        # if self.conv_layer == 'GCN2Conv':
        #     x_0 = x

        for conv in self.conv:
            # if self.conv_layer == 'GCN2Conv':
            #     x = tanh(conv(x, x_0, edge_index))
            # else:
            # x = F.relu(conv(x, edge_index))
            x = tanh(conv(x, edge_index))
            # x = nn.BatchNorm1d(x.size(1)).to(device)(x)
        if not self.node_level_task:
            x = torch.cat([global_add_pool(x, batch), global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)
        x = self.readout(x)

        return x
