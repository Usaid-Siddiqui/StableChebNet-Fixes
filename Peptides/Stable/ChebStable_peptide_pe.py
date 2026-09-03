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
from utils import pe,eval_ap
from utils import *
from model_euler import EulerModel


# Load JSON config
with open('./config_StableCheb.json', 'r') as f:  #### Change path accordingly
    config = json.load(f)

# Define ArgumentParser
parser = argparse.ArgumentParser()
parser.add_argument('--hidden', type=int, default=config.get("hidden"))
parser.add_argument('--seed', type=int, default=config.get("seed"))
parser.add_argument('--batch_size', type=int, default=config.get("batch_size"))
parser.add_argument('--K', type=int, default=config.get("K"))
parser.add_argument('--num_layers', type=int, default=config.get("num_layers"))
parser.add_argument('--mlp_layers', type=int, default=config.get("mlp_layers"))
parser.add_argument('--step_size', type=float, default=config.get("step_size"))
parser.add_argument('--dissipative_force', type=float, default=config.get("dissipative_force"))
parser.add_argument('--lr', type=float, default=config.get("lr"))
parser.add_argument('--epochs', type=int, default=config.get("epochs"))
parser.add_argument('--pos_enc', type=str, choices=['laplacian','random_walk'],
                     default=config.get("pos_enc","laplacian"),
                     help="Which positional encoding to add")
parser.add_argument('--pe_dim', type=int, default=config.get("pe_dim", 8),
                     help="Dimension (k) of the positional encoding")
args = parser.parse_args()


from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
from torch_geometric.data import Data

# this wrapper will never request k >= N-1, avoiding the scipy error
class SafeLaplacianPE:
    def __init__(self, k, attr_name=None):
        self.k = k
        self.attr = attr_name
    def __call__(self, data: Data):
        N = data.num_nodes
        # if graph is too small, skip entirely
        k_eff = min(self.k, max(0, N - 2))
        if k_eff > 0:
            return AddLaplacianEigenvectorPE(k=k_eff, attr_name=self.attr)(data)
        else:
            return data
        
from torch_geometric.transforms import BaseTransform, AddLaplacianEigenvectorPE

class SafeAddLapPE(BaseTransform):
    """Like AddLaplacianEigenvectorPE but never crashes when k ≥ N‑1.
    · If k_eff (= min(k, N‑2)) == 0 we skip the eigendecomposition and
      return an all‑zero PE tensor of shape [N, k].
    · Otherwise we compute with k_eff and right‑pad with zeros so that
      every graph still outputs *exactly* `k` channels.
    This is the clipping logic the original LapPE authors used in their
    Peptides‑func code release.
    """

    def __init__(self, k: int = 16, attr_name: str = 'lap_pe'):
        self.k = k
        self.attr_name = attr_name

    def __call__(self, data):
        import torch
        N = int(data.num_nodes)
        # ➊ choose a legal k_eff < N‑1 (can be zero!)
        k_eff = max(0, min(self.k, N - 2))

        if k_eff > 0:
            # compute the eigenvectors safely
            data = AddLaplacianEigenvectorPE(k=k_eff, attr_name=self.attr_name)(data)
            # right‑pad if we could not get the full k
            if k_eff < self.k:
                pad = data[self.attr_name].new_zeros(N, self.k - k_eff)
                data[self.attr_name] = torch.cat([data[self.attr_name], pad], dim=-1)
        else:
            # N ≤ 2 ⇒ no valid non‑trivial eigenvector exists – use zeros
            data[self.attr_name] = torch.zeros((N, self.k), dtype=torch.float32)

        return data

from torch_geometric.transforms import AddLaplacianEigenvectorPE
# check=AddLaplacianEigenvectorPE(k=8)
torch.manual_seed(args.seed)

# 1) Define a transform that pads any graph with N<16 up to exactly 16 nodes:
class PadToMinNodes(object):
    def __init__(self, min_nodes: int = 16):
        self.min_nodes = min_nodes

    def __call__(self, data):
        N, Fdim = data.num_nodes, data.x.size(1)
        if N < self.min_nodes:
            # create (min_nodes-N) zero–feature nodes
            pad_x = data.x.new_zeros((self.min_nodes - N, Fdim))
            data.x = torch.cat([data.x, pad_x], dim=0)
            # since these are isolated, we leave edge_index unchanged
            # update metadata
            # data.num_nodes = self.min_nodes
        return data
    
from torch_geometric.transforms import Compose, AddLaplacianEigenvectorPE
# 2) Compose padding + Laplacian PE (k=16)
transform = Compose([
    PadToMinNodes(min_nodes=17),
    AddLaplacianEigenvectorPE(k=16, attr_name='lap_pe', is_undirected=True),
])
if args.pos_enc == 'laplacian':
    tf = transform#SafeAddLapPE(k=args.pe_dim, attr_name='lap_pe')

elif args.pos_enc == 'random_walk':
    # Append k random-walk features to data.x
    tf = AddRandomWalkPE(walk_length=args.pe_dim, attr_name='lap_pe')
else:
    tf = None

my_dataset='Peptides-func'
dataset1 = LRGBDataset(root='./', name=my_dataset, transform=tf, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes= 10 #dataset1.num_classes


from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=args.batch_size, shuffle=False)

model = EulerModel(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes,args.step_size,args.dissipative_force).to(device)

from torch.optim import AdamW
# --- Optimizer ---------------------------------------------------------
optimizer = AdamW(
    model.parameters(),
    lr=1e-3,               # base lr
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
)

criterion = torch.nn.CrossEntropyLoss()

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {total_params}")

wandb.init(
project= "Peptide_Compare",#"PeptideFunc2025",
name="ChebEuler_"+str(args.K)+"_PE",
config=config,
)

wandb.log({"Params": total_params})

# Optionally, log the args (if modified via command line)
wandb.config.update(args)

import datetime
today_date=datetime.datetime.now()
## hour and minute 
today_date=today_date.strftime("%Y-%m-%d-%H-%M")

checkpoint_path='./SmartRewire/ChebNet_Baseline/temp_weights/best_epoch_'+str(today_date)+'.pth'

temp=0
for epoch in range(args.epochs):
  print(torch.version.cuda)

  model.train()
  correct = 0
  precision=0
  real=[]
  pred=[]
  for i, data in enumerate(trainloader):

    data=data.to(device)

    optimizer.zero_grad()

    classify=model(data.x, data.edge_index, data.batch, device,data)

    loss = criterion(classify, data.y)  # Compute the loss

    loss.backward()

    optimizer.step()

    real.append(data.y)
    pred.append(classify)

  y_true = torch.cat(real, dim=0)
  y_pred = torch.cat(pred, dim=0)
  train_perf = eval_ap(y_true=y_true, y_pred=y_pred)
  del real
  del pred


  train_acc=precision / (i+1)
  # if epoch >=40==0:
  # if epoch %40==0:
  #   optimizer.param_groups[0]["lr"]=optimizer.param_groups[0]["lr"]*0.9
  # scheduler.step()


  val_correct=0
  val_precision=0
  valreal=[]
  valpred=[]
  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    val_classify=model(valdata.x, valdata.edge_index, valdata.batch, device,valdata)

    val_loss = criterion(val_classify, valdata.y)
    val_pred = val_classify.argmax(dim=1)
    val_precision+=eval_ap(valdata.y,val_classify)
    valreal.append(valdata.y)
    valpred.append(val_classify)

  val_y_true = torch.cat(valreal, dim=0)
  val_y_pred = torch.cat(valpred, dim=0)
  val_perf = eval_ap(y_true=val_y_true, y_pred=val_y_pred)
  scheduler.step(val_loss)

  if val_perf>=temp:
    temp=val_perf
    when=epoch
    # torch.save(model.state_dict(), checkpoint_path)


  print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f},Train Acc: {train_perf:.4f}, Val_Loss: {val_loss.item():.4f},Val Acc: {val_perf:.4f}')
  wandb.log({"Train Acc": train_perf})
  wandb.log({"Val Acc": val_perf})
  wandb.log({"Train Loss": loss})
  wandb.log({"Val Loss": val_loss})
  wandb.log({"Epoch": epoch})

device="cuda"
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint)

test_precision=0
tr=[]
tp=[]
with torch.no_grad():
  for k, testdata in enumerate(testloader):
    model.eval()
    model=model.to(device)
    testdata=testdata.to(device)

    test_classify=model(testdata.x, testdata.edge_index, testdata.batch, device,testdata)

    tr.append(testdata.y)
    tp.append(test_classify)

  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)

  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)
  test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)

wandb.log({"Test Acc": test_perf})
wandb.finish()