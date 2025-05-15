import torch
from torch_geometric.datasets import LRGBDataset
import os.path as osp
import torch_geometric.transforms as T
import wandb
from torch_geometric.loader import DataLoader
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
import json
# from utils import pe,eval_ap
from utils import *

from HNOStruc import HNOStruc


# Load JSON config
with open('config_struc.json', 'r') as f:   #### Change path accordingly
    config = json.load(f)

# Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=config.get("hidden"))
parser.add_argument('--seed', type=int, default=config.get("seed"))
parser.add_argument('--batch_size', type=int, default=config.get("batch_size"))
parser.add_argument('--K', type=int, default=config.get("K"))
parser.add_argument('--num_layers', type=int, default=config.get("num_layers"))
parser.add_argument('--mlp_layers', type=int, default=config.get("mlp_layers"))
parser.add_argument('--lr', type=float, default=config.get("lr"))
parser.add_argument('--epochs', type=int, default=config.get("epochs"))
args = parser.parse_args()

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=8)
torch.manual_seed(args.seed)

# if args.laplace_RW==True:
#   from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
#   tf=AddRandomWalkPE(walk_length=8, attr_name='rwe')
# else:
#    tf=None

tf=None
my_dataset='Peptides-struct'
dataset1 = LRGBDataset(root='./', name=my_dataset, transform=tf, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes


# if args.laplace_RW==True:
#   dataset1 = pe(dataset1)
#   validation_set1 = pe(validation_set1)
#   test_set1 = pe(test_set1)



from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=args.batch_size, shuffle=False)


model = HNOStruc(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes).to(device)
from torch.optim import AdamW
# --- Optimizer ---------------------------------------------------------
optimizer = AdamW(
    model.parameters(),
    lr=args.lr,               # base lr
    betas=(0.9, 0.999),    # AdamW’s analogue of “momentum = 0.9”
    weight_decay=0.0       # set if you use weight‑decay regularisation
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

# --- Reduce‑on‑Plateau scheduler --------------------------------------
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",            # “min” because we’ll watch validation loss
    factor=0.5,            # reduce_factor
    patience=20,           # schedule_patience for Peptides
    min_lr=1e-5,           # min_lr
    verbose=True           # logs each lr change
)

criterion = torch.nn.CrossEntropyLoss()

wandb.init(
project="PeptideStruc2025",
name="Cheb_"+str(args.K),
config=config,
)

# Optionally, log the args (if modified via command line)
wandb.config.update(args,allow_val_change=True)


criterion = torch.nn.L1Loss()

temp=10000000
for epoch in range(args.epochs):
  model.train()
  correct = 0
  totalLoss=0
  total_loss = 0
  N = 0

  for i, data in enumerate(trainloader):

    data=data.to(device)

    optimizer.zero_grad()

    classify=model(data.x, data.edge_index, data.batch, device)

    mask = ~torch.isnan(data.y)

    loss = (classify[mask].squeeze() - data.y[mask]).abs().mean() 

    loss.backward()

    total_loss += loss.item() * data.num_graphs

    N += data.num_graphs

    optimizer.step()

    totalLoss+=loss

  totalLoss=totalLoss / (i+1)

  train_loss = total_loss / N
  train_perf = train_loss
#   scheduler.step()

  val_correct=0
  val_precision=0
  val_correct=0
  #totalVaLoss=0
  total_val_loss=0
  Nval=0

  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    val_classify=model(valdata.x, valdata.edge_index, valdata.batch, device)

    # val_loss = criterion(val_classify, valdata.y)
    valmask = ~torch.isnan(valdata.y)

    # val_loss = criterion(val_classify, valdata.y)
    val_loss=(val_classify[valmask].squeeze() -valdata.y[valmask]).abs().mean()

    total_val_loss += val_loss.item()*valdata.num_graphs
    Nval += valdata.num_graphs

  Val_loss = total_val_loss/Nval
  val_perf = Val_loss

  # if val_perf<temp:
  #   temp=val_perf
  #   when=epoch
  #   torch.save(model.state_dict(), checkpoint_path)


  print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f},Train Acc: {train_perf:.4f}, Val_Loss: {val_loss.item():.4f},Val Acc: {val_perf:.4f}')
  wandb.log({"train perf": train_perf})
  wandb.log({"Val perf": val_perf})
  wandb.log({"Epoch": epoch})

device="cuda"
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint)


totalTest=0

test_precision=0
total_test_loss=0
Ntest=0
with torch.no_grad():
  for k, testdata in enumerate(testloader):
    model.eval()
    model=model.to(device)
    testdata=testdata.to(device)

    test_classify=model(testdata.x, testdata.edge_index, testdata.batch, device)

    testmask = ~torch.isnan(testdata.y)

    testloss = (test_classify[testmask].squeeze() - testdata.y[testmask]).abs().mean()
    total_test_loss += testloss.item()*testdata.num_graphs
    Ntest += testdata.num_graphs
test_loss = total_test_loss/Ntest
test_perf = -test_loss
wandb.log({"Test Loss": test_loss})
wandb.log({"Test perf": test_perf})
wandb.finish()