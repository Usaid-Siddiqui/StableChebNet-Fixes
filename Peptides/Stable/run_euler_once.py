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
with open('./config_StableCheb.json', 'r') as f:
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
args = parser.parse_args()

from torch_geometric.transforms import AddLaplacianEigenvectorPE
check=AddLaplacianEigenvectorPE(k=8)
torch.manual_seed(args.seed)

tf=None
my_dataset='Peptides-func'
dataset1 = LRGBDataset(root='./', name=my_dataset, transform=tf, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes


from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=args.batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=args.batch_size, shuffle=False)

model = EulerModel(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes,args.step_size,args.dissipative_force).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

criterion = torch.nn.CrossEntropyLoss()

# Count the total number of parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {total_params}")

wandb.init(
project="PeptideFunc2025",
name="FullChebEuler_"+str(args.K)+"",
config=config,
)

wandb.log({"Params": total_params})

# Optionally, log the args (if modified via command line)
wandb.config.update(args)

import torch.optim as optim
from torch.optim import Adagrad, AdamW, Optimizer
import torch_geometric.graphgym.register as register
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler

scheduler=cosine_with_warmup_scheduler(optimizer,num_warmup_epochs=5,max_epoch=300)


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

    classify=model(data.x, data.edge_index, data.batch, device)

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
  scheduler.step()


  val_correct=0
  val_precision=0
  valreal=[]
  valpred=[]
  for j, valdata in enumerate(valoader):
    model.eval()
    valdata=valdata.to(device)

    val_classify=model(valdata.x, valdata.edge_index, valdata.batch, device)

    val_loss = criterion(val_classify, valdata.y)
    val_pred = val_classify.argmax(dim=1)
    val_precision+=eval_ap(valdata.y,val_classify)
    valreal.append(valdata.y)
    valpred.append(val_classify)

  val_y_true = torch.cat(valreal, dim=0)
  val_y_pred = torch.cat(valpred, dim=0)
  val_perf = eval_ap(y_true=val_y_true, y_pred=val_y_pred)
  # scheduler.step()

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

    test_classify=model(testdata.x, testdata.edge_index, testdata.batch, device)

    tr.append(testdata.y)
    tp.append(test_classify)

  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)

  y_preds = torch.cat(tp, dim=0)
  y_trues = torch.cat(tr, dim=0)
  test_perf = eval_ap(y_true=y_trues, y_pred=y_preds)

wandb.log({"Test Acc": test_perf})
wandb.finish()