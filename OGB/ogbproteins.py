
# Install required packages.
import os
import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm
import wandb 
import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.loader import RandomNodeLoader
from torch_geometric.nn import DeepGCNLayer, GENConv,ChebConv
from torch_geometric.utils import scatter
### Seed everything
import random
seed = 1000
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

dataset = PygNodePropPredDataset('ogbn-proteins', root='../data')
splitted_idx = dataset.get_idx_split()
data = dataset[0]
data.node_species = None
data.y = data.y.to(torch.float)

# Initialize features of nodes by aggregating edge features.
row, col = data.edge_index
data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')

# Set split indices to masks.
for split in ['train', 'valid', 'test']:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[splitted_idx[split]] = True
    data[f'{split}_mask'] = mask

train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True,num_workers=5)
test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), 1)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

class DeeperCheb(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers):
        super().__init__()

        self.node_encoder = Linear(data.x.size(-1), hidden_channels)
        self.edge_encoder = Linear(data.edge_attr.size(-1), 1)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = ChebConv(hidden_channels, hidden_channels,K=16)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, data.y.size(-1))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)



BN = True

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def reset_parameters(self):
        pass


# from torch_scatter import scatter
class MLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True):
        super().__init__()
        n_hid = nin
        self.layers = nn.ModuleList([nn.Linear(nin if i == 0 else n_hid,
                                     n_hid if i < nlayer-1 else nout,
                                     # TODO: revise later
                                               bias=True if (i == nlayer-1 and not with_final_activation and bias)
                                               or (not with_norm) else False)  # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(n_hid if i < nlayer-1 else nout) if with_norm else Identity()
                                    for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin == nout)  # TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x):
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x)
                x = F.relu(x)

        # if self.residual:
        #     x = x + previous_x
        return x

from torch import nn, einsum
import torch.nn.functional as F


K=8
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
class HNO(torch.nn.Module):

    def __init__(self,hidden_channels):
        super(HNO, self).__init__()

        self.conv1 = ChebConv(data.x.size(-1), hidden_channels,K=K)
        self.conv2 = ChebConv(hidden_channels, hidden_channels,K=K)
        self.conv3 = ChebConv(hidden_channels, hidden_channels,K=K)
        self.conv4 = ChebConv(hidden_channels, hidden_channels,K=K)


        self.bano1 = torch.nn.BatchNorm1d(num_features= int(hidden_channels))
        self.bano2 = torch.nn.BatchNorm1d(num_features= int(hidden_channels))
        self.bano3 = torch.nn.BatchNorm1d(num_features= int(hidden_channels))


        # self.mlpRep = MLP(int(hidden_channels), data.y.size(-1), nlayer=2, with_final_activation=True)
        self.mlpRep = nn.Linear(int(hidden_channels), data.y.size(-1))
    def forward(self, x1, edge_index):


        x = self.conv1(x1, edge_index)
        x = F.tanh(x)
        # x=self.bano1(x)
        x = F.dropout(x, training=self.training,p=0.1)

        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        # x=self.bano2(x)
        x = F.dropout(x, training=self.training,p=0.1)

        x = self.conv3(x, edge_index)
        x = F.tanh(x)
        # x = self.bano3(x)
        x = F.dropout(x, training=self.training,p=0.1)

        x = self.conv4(x, edge_index)
        x = F.tanh(x)
        # x = self.bano3(x)
        x = F.dropout(x, training=self.training,p=0.1)


        x=self.mlpRep(x)

        return x

learning_rate=0.001
hidden_dims=55
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HNO(hidden_channels=hidden_dims).to(device)
# model = DeeperGCN(hidden_channels=hidden_dims, num_layers=28).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.BCEWithLogitsLoss()
evaluator = Evaluator('ogbn-proteins')

chosen_model='HNO'
# chosen_model='Deeper'

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# chosen_model='HNO'
print(f"Number of trainable parameters: {total_params}")



def train(epoch):
    model.train()

    pbar = tqdm(total=len(train_loader))
    pbar.set_description(f'Training epoch: {epoch:04d}')

    total_loss = total_examples = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        if chosen_model=='HNO':
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * int(data.train_mask.sum())
        total_examples += int(data.train_mask.sum())

        pbar.update(1)

    pbar.close()

    return total_loss / total_examples

@torch.no_grad()
def test():
    model.eval()

    y_true = {'train': [], 'valid': [], 'test': []}
    y_pred = {'train': [], 'valid': [], 'test': []}

    pbar = tqdm(total=len(test_loader))
    pbar.set_description(f'Evaluating epoch: {epoch:04d}')

    for data in test_loader:
        data = data.to(device)
        if chosen_model=='HNO':
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, data.edge_attr)

        for split in y_true.keys():
            mask = data[f'{split}_mask']
            y_true[split].append(data.y[mask].cpu())
            y_pred[split].append(out[mask].cpu())

        pbar.update(1)

    pbar.close()

    train_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['train'], dim=0),
        'y_pred': torch.cat(y_pred['train'], dim=0),
    })['rocauc']

    valid_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['valid'], dim=0),
        'y_pred': torch.cat(y_pred['valid'], dim=0),
    })['rocauc']

    test_rocauc = evaluator.eval({
        'y_true': torch.cat(y_true['test'], dim=0),
        'y_pred': torch.cat(y_pred['test'], dim=0),
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

config = dict (
lr=learning_rate,
hidden=hidden_dims,
K=K,
)

wandb.init(
project="OgbProteins_2025",
name="DeepCheb",
config=config,
)

wandb.log({"Params": total_params})
for epoch in range(1, 350):
    loss = train(epoch)
    train_rocauc, valid_rocauc, test_rocauc = test()
    print(f'Loss: {loss:.4f}, Train: {train_rocauc:.4f}, '
          f'Val: {valid_rocauc:.4f}, Test: {test_rocauc:.4f}')
    wandb.log({"Train ROC": train_rocauc})
    wandb.log({"Val ROC": valid_rocauc})
    wandb.log({"Train Loss": loss})
    wandb.log({"Epoch": epoch})
    wandb.log({"Test ROC": test_rocauc})
