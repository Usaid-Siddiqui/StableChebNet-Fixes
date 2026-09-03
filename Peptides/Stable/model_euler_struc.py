from torch import nn, einsum
import torch.nn.functional as F
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv,global_mean_pool, ChebConv,global_add_pool,global_max_pool
import torch
from MLP import MLP
from torch.nn import Linear
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
from EulerConv import Euler_ChebConv

import json
# Load JSON config
with open('./config_StableChebStruc.json', 'r') as f:
    config = json.load(f)

### Get pe type
pe_type = config.get("pos_enc", "laplacian")

# if pe_type == 'laplacian':
#     dim= 9+config.get("pe_dim", 8)
# else:
#     dim= 9
dim=9
class EulerModelstruc(torch.nn.Module):

    def __init__(self,hidden_dims,K,num_layers,mlp_layers,num_classes,step_size,dissipative_force,damping_kernel='dirichlet'):
        super(EulerModelstruc, self).__init__()

        self.lin=Linear(dim,hidden_dims)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers): #- 1):
            self.convs.append(Euler_ChebConv(hidden_dims, hidden_dims,K,step_size,dissipative_force,damping_kernel=damping_kernel))

        self.bano1 = torch.nn.BatchNorm1d(num_features= hidden_dims)
        self.bano2 = torch.nn.BatchNorm1d(num_features= hidden_dims)
        self.bano3 = torch.nn.BatchNorm1d(num_features= hidden_dims)

        self.mlpRep = MLP(hidden_dims, num_classes, nlayer=mlp_layers, with_final_activation=False)
        
        # ---------- fixed dropout rates ----------
        self.drop_probs = [0.25, 0.20] + [0.15] * (num_layers - 2)

    def forward(self, x, edge_index, batch, device):  
        x=x.float()
        
        x=self.lin(x)
        i=0
        for conv in self.convs:
            x = conv(x, edge_index)
            ## if not last layer
            # if conv != self.convs[-1]:
            x = F.silu(x)
            x = nn.BatchNorm1d(x.size(1)).to(device)(x)
            # x = F.dropout(x, p=0.15, training=self.training)
            x = F.dropout(x,
                          p=self.drop_probs[i],   # 0.25 → 0.20 → 0.15 …
                          training=self.training)

        final = global_add_pool(x, batch)

        classifier=self.mlpRep(final)

        return classifier
