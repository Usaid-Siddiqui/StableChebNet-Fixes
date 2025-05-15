from torch import nn, einsum
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool, ChebConv,global_add_pool
import torch
from MLP import MLP
from torch.nn import Linear
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data

class HNOStruc(torch.nn.Module):

    def __init__(self,hidden_dims,K,num_layers,mlp_layers,num_classes):
        super(HNOStruc, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(ChebConv(9, hidden_dims,K=K))
        for _ in range(num_layers - 1):
            self.convs.append(ChebConv(hidden_dims, hidden_dims,K=K))

        self.bano1 = torch.nn.BatchNorm1d(num_features= hidden_dims)
        self.bano2 = torch.nn.BatchNorm1d(num_features= hidden_dims)
        self.bano3 = torch.nn.BatchNorm1d(num_features= hidden_dims)

        self.mlpRep = MLP(hidden_dims, num_classes, nlayer=mlp_layers, with_final_activation=False)
        #self.mlpRep2 = MLP(int(hidden_dims), num_classes, nlayer=2, with_final_activation=False)

    def forward(self, x, edge_index, batch, device):
        x=x.float()
        m=0
        for conv in self.convs:
        ###all except last
            # if m<(len(self.convs)-1):
            x = conv(x, edge_index)
            x = F.leaky_relu(x)
            x = nn.BatchNorm1d(x.size(1)).to(device)(x)
            x = F.dropout(x, p=0.1, training=self.training)
            m+=1

        final = global_add_pool(x, batch)

        classifier=self.mlpRep(final)

        return classifier
