import torch
from torch_geometric.datasets import LRGBDataset
import os
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
import json
# from utils import pe,eval_ap
from utils import *

# from HNOStruc import HNOStruc
from model_euler_struc import EulerModelstruc
from experiment_utils import get_wandb, probe_model, probe_epochs, maybe_subset, append_csv

wandb = get_wandb()

# Load JSON config
with open('./config_StableChebStruc.json', 'r') as f: #### Change path accordingly
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
parser.add_argument('--damping_kernel', type=str, default=config.get("damping_kernel", "dirichlet"),
                    choices=['dirichlet', 'uniform', 'fejer'])
parser.add_argument('--lr', type=float, default=config.get("lr"))
parser.add_argument('--epochs', type=int, default=config.get("epochs"))
parser.add_argument('--pos_enc', type=str, choices=['laplacian','random_walk'],
                     default=config.get("pos_enc","laplacian"),
                     help="Which positional encoding to add")
parser.add_argument('--pe_dim', type=int, default=config.get("pe_dim", 8),
                     help="Dimension (k) of the positional encoding")
# ---- experiment plumbing ----
parser.add_argument('--max_graphs', type=int, default=None,
                    help='subset train/val/test to this many graphs (dry run only)')
parser.add_argument('--results_csv', type=str, default=os.environ.get("RESULTS_CSV", ""))
parser.add_argument('--run_name', type=str, default="")
parser.add_argument('--wandb_project', type=str, default=os.environ.get("WANDB_PROJECT", "PeptideStruc2025"))
parser.add_argument('--no_spectral', action='store_true')
parser.add_argument('--ckpt_dir', type=str, default=os.environ.get("CKPT_DIR", "checkpoints"))
parser.add_argument('--no_checkpoint', action='store_true')
args = parser.parse_args()


from torch_geometric.transforms import AddLaplacianEigenvectorPE
# check=AddLaplacianEigenvectorPE(k=8)
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

dataset1 = maybe_subset(dataset1, args.max_graphs)
validation_set1 = maybe_subset(validation_set1, args.max_graphs)
test_set1 = maybe_subset(test_set1, args.max_graphs)

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes


from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=args.batch_size, shuffle=False)

model = EulerModelstruc(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes,args.step_size,args.dissipative_force,damping_kernel=args.damping_kernel).to(device)
# model = HNOStruc(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes).to(device)
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)


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
)


criterion = torch.nn.CrossEntropyLoss()

_run_name = args.run_name or f"{args.damping_kernel}_g{args.dissipative_force}_s{args.seed}_K{args.K}"
wandb.init(
project=args.wandb_project,
name=_run_name,
config=config,
)

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {total_params}")

wandb.log({"Params": total_params})

# Optionally, log the args (if modified via command line)
wandb.config.update(args,allow_val_change=True)

criterion = torch.nn.L1Loss()

# ---- fixed batch for the ||J||_2 spectral probe ----
_probe_epochs = set() if args.no_spectral else set(probe_epochs(args.epochs))
_probe_batch = next(iter(DataLoader(dataset1, batch_size=args.batch_size, shuffle=False))) if _probe_epochs else None
_spectral_csv = os.path.splitext(args.results_csv)[0] + "_spectral.csv" if args.results_csv else ""

os.makedirs(args.ckpt_dir, exist_ok=True)
checkpoint_path = os.path.join(args.ckpt_dir, _run_name + ".pth")

temp=10000000
when=0
diverged=False
diverged_epoch=-1
for epoch in range(args.epochs):

  # ---- ||J||_2 spectral probe at selected epochs ----
  if epoch in _probe_epochs:
      try:
          jn = probe_model(model, _probe_batch, device, args.hidden)
          for li, val in enumerate(jn):
              wandb.log({f"specnorm/layer{li}": val, "Epoch": epoch})
          wandb.log({"specnorm/max": max(jn), "Epoch": epoch})
          print(f"[probe] epoch {epoch:03d}  ||J||_2 per layer: "
                + ", ".join(f"{v:.4f}" for v in jn) + f"   (max {max(jn):.4f})")
          if _spectral_csv:
              append_csv(_spectral_csv, {"run": _run_name, "kernel": args.damping_kernel,
                  "gamma": args.dissipative_force, "seed": args.seed, "epoch": epoch,
                  **{f"layer{li}": v for li, v in enumerate(jn)}, "max_layer": max(jn)})
      except Exception as e:
          print(f"[probe] epoch {epoch} failed: {e}")

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

    if not torch.isfinite(loss):
        print(f"[diverged] non-finite training loss at epoch {epoch}; stopping early "
              f"(best-val epoch {when}, val {temp:.4f})")
        diverged = True; diverged_epoch = epoch
        break

    loss.backward()

    total_loss += loss.item() * data.num_graphs

    N += data.num_graphs

    optimizer.step()

    totalLoss+=loss

  if diverged:
    break

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
  scheduler.step(Val_loss)

  if val_perf<temp:
    temp=val_perf
    when=epoch
    if not args.no_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)


  print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f},Train Acc: {train_perf:.4f}, Val_Loss: {val_loss.item():.4f},Val Acc: {val_perf:.4f}')
  wandb.log({"train perf": train_perf})
  wandb.log({"Val perf": val_perf})
  wandb.log({"Epoch": epoch})

# load the best-val checkpoint for the test eval (standard protocol, not final-epoch)
if not args.no_checkpoint and os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print(f"[checkpoint] loaded best-val model from {checkpoint_path} (epoch {when}, val {temp:.4f})")


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

print(f"[result] kernel={args.damping_kernel} gamma={args.dissipative_force} seed={args.seed} "
      f"K={args.K} eps={args.step_size} layers={args.num_layers}  test_MAE={float(test_loss):.4f}  "
      f"best_val_MAE={float(temp):.4f} @epoch {when}  diverged={diverged}"
      + (f" @epoch {diverged_epoch}" if diverged else ""))
if args.results_csv:
    append_csv(args.results_csv, {
        "run": _run_name, "kernel": args.damping_kernel, "gamma": args.dissipative_force,
        "seed": args.seed, "K": args.K, "eps": args.step_size, "hidden": args.hidden,
        "num_layers": args.num_layers, "epochs": args.epochs,
        "test_MAE": float(test_loss), "best_val_MAE": float(temp), "best_epoch": int(when),
        "diverged": int(diverged), "diverged_epoch": int(diverged_epoch),
    })

wandb.finish()