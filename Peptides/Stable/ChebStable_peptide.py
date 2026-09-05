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
from utils import pe,eval_ap
from utils import *
from model_euler import EulerModel
from experiment_utils import get_wandb, probe_model, probe_epochs, maybe_subset, append_csv

wandb = get_wandb()


# Load JSON config
with open('./config_StableCheb.json', 'r') as f:    #### Change path accordingly
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
# ---- experiment plumbing (do not affect the paper's default path) ----
parser.add_argument('--max_graphs', type=int, default=None,
                    help='subset train/val/test to this many graphs (dry run only)')
parser.add_argument('--results_csv', type=str, default=os.environ.get("RESULTS_CSV", ""),
                    help='append the final result row to this CSV')
parser.add_argument('--run_name', type=str, default="", help='wandb run name / CSV tag')
parser.add_argument('--wandb_project', type=str, default=os.environ.get("WANDB_PROJECT", "Peptide_Compare"))
parser.add_argument('--no_spectral', action='store_true', help='disable the ||J||_2 probe')
parser.add_argument('--ckpt_dir', type=str, default=os.environ.get("CKPT_DIR", "checkpoints"),
                    help='directory for best-val checkpoints (one .pth per run)')
parser.add_argument('--no_checkpoint', action='store_true', help='disable checkpoint save/load')
args = parser.parse_args()

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=8)
torch.manual_seed(args.seed)

tf=None
my_dataset='Peptides-func'
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

model = EulerModel(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes,args.step_size,args.dissipative_force,damping_kernel=args.damping_kernel).to(device)

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

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {total_params}")

_run_name = args.run_name or f"{args.damping_kernel}_g{args.dissipative_force}_s{args.seed}_K{args.K}"
wandb.init(
project= args.wandb_project,
name=_run_name,
config=config,
)

wandb.log({"Params": total_params})

# Optionally, log the args (if modified via command line)
wandb.config.update(args,allow_val_change=True)

# ---- fixed batch for the ||J||_2 spectral probe (section 13) ----
_probe_epochs = set() if args.no_spectral else set(probe_epochs(args.epochs))
_probe_batch = next(iter(DataLoader(dataset1, batch_size=args.batch_size, shuffle=False))) if _probe_epochs else None
_spectral_csv = os.path.splitext(args.results_csv)[0] + "_spectral.csv" if args.results_csv else ""

# import torch.optim as optim
# from torch.optim import Adagrad, AdamW, Optimizer
# import torch_geometric.graphgym.register as register
# def get_cosine_schedule_with_warmup(
#         optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
#         num_cycles: float = 0.5, last_epoch: int = -1):

#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

#     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


# @register.register_scheduler('cosine_with_warmup')
# def cosine_with_warmup_scheduler(optimizer: Optimizer,
#                                  num_warmup_epochs: int, max_epoch: int):
#     scheduler = get_cosine_schedule_with_warmup(
#         optimizer=optimizer,
#         num_warmup_steps=num_warmup_epochs,
#         num_training_steps=max_epoch
#     )
#     return scheduler

# scheduler=cosine_with_warmup_scheduler(optimizer,num_warmup_epochs=5,max_epoch=300)


os.makedirs(args.ckpt_dir, exist_ok=True)
checkpoint_path = os.path.join(args.ckpt_dir, _run_name + ".pth")

temp=0
when=0
diverged=False
diverged_epoch=-1
for epoch in range(args.epochs):

  # ---- ||J||_2 spectral probe at selected epochs (theory <-> training link) ----
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
  precision=0
  real=[]
  pred=[]
  for i, data in enumerate(trainloader):

    data=data.to(device)

    optimizer.zero_grad()

    classify=model(data.x, data.edge_index, data.batch, device,data)

    loss = criterion(classify, data.y)  # Compute the loss

    if not torch.isfinite(loss):
        # forward Euler diverged (expected at depth); stop cleanly, keep best-val ckpt
        print(f"[diverged] non-finite training loss at epoch {epoch}; stopping early "
              f"(best-val epoch {when}, val {temp:.4f})")
        diverged = True; diverged_epoch = epoch
        break

    loss.backward()

    optimizer.step()

    real.append(data.y)
    pred.append(classify)

  if diverged:
    break

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
    if not args.no_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)


  print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f},Train Acc: {train_perf:.4f}, Val_Loss: {val_loss.item():.4f},Val Acc: {val_perf:.4f}')
  wandb.log({"Train Acc": train_perf})
  wandb.log({"Val Acc": val_perf})
  wandb.log({"Train Loss": loss})
  wandb.log({"Val Loss": val_loss})
  wandb.log({"Epoch": epoch})

# load the best-val checkpoint for the test eval (standard protocol, not final-epoch)
if not args.no_checkpoint and os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    print(f"[checkpoint] loaded best-val model from {checkpoint_path} (epoch {when}, val {temp:.4f})")

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
  try:
      test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)
  except ValueError:
      # no usable (finite) checkpoint -- run diverged before any val improvement
      test_perf = float('nan')

wandb.log({"Test Acc": test_perf})

print(f"[result] kernel={args.damping_kernel} gamma={args.dissipative_force} seed={args.seed} "
      f"K={args.K} eps={args.step_size} layers={args.num_layers}  test_AP={float(test_perf):.4f}  "
      f"best_val_AP={float(temp):.4f} @epoch {when}  diverged={diverged}"
      + (f" @epoch {diverged_epoch}" if diverged else ""))
if args.results_csv:
    append_csv(args.results_csv, {
        "run": _run_name, "kernel": args.damping_kernel, "gamma": args.dissipative_force,
        "seed": args.seed, "K": args.K, "eps": args.step_size, "hidden": args.hidden,
        "num_layers": args.num_layers, "epochs": args.epochs,
        "test_AP": float(test_perf), "best_val_AP": float(temp), "best_epoch": int(when),
        "diverged": int(diverged), "diverged_epoch": int(diverged_epoch),
    })

wandb.finish()