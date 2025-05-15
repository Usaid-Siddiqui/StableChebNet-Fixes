import torch
import datetime
import wandb
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE
import json
from utils import pe,eval_ap
from utils import *
from model_euler import EulerModel
import random


batch_size=32
lr=0.0001


tf=None
my_dataset='Peptides-func'
dataset1 = LRGBDataset(root='./', name=my_dataset, transform=tf, split="train")#.shuffle()
validation_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="val")#.shuffle()
test_set1 = LRGBDataset(root='./', name=my_dataset,transform=tf, split="test")#.shuffle()

num_feats=dataset1.num_node_features
num_classes=dataset1.num_classes

from torch_geometric.loader import DataLoader
trainloader = DataLoader(dataset1, batch_size=batch_size, shuffle=True,drop_last=False)
valoader = DataLoader(validation_set1, batch_size=batch_size, shuffle=False)
testloader = DataLoader(test_set1, batch_size=batch_size, shuffle=False)

criterion = torch.nn.CrossEntropyLoss()



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



def run_entire_pipeline(
    model,
    seed,
    trainloader,
    valoader,
    testloader,
    device,
    optimizer,
    scheduler,
    criterion,
    eval_ap,
):
    """
    Runs the entire training, validation, and testing pipeline in one function call.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Optionally log the args (in case they've been modified via command line)
    # wandb.config.update(args)

    # Generate a time-stamped checkpoint name
    today_date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    checkpoint_path = "./SmartRewire/ChebNet_Baseline/temp_weights/best_epoch_" + str(today_date) + ".pth"

    best_val_perf = 0.0
    when = -1  # Track which epoch gave the best performance

    # -------------------------
    # TRAINING & VALIDATION
    # -------------------------
    for epoch in range(250):
        model.train()
        real_train = []
        pred_train = []

        # -------------------------
        # TRAINING LOOP
        # -------------------------
        for i, data in enumerate(trainloader):
            data = data.to(device)

            optimizer.zero_grad()

            classify = model(data.x, data.edge_index, data.batch, device)
            loss = criterion(classify, data.y)
            loss.backward()
            optimizer.step()

            real_train.append(data.y)
            pred_train.append(classify)

        # Compute training performance
        y_true = torch.cat(real_train, dim=0)
        y_pred = torch.cat(pred_train, dim=0)
        train_perf = eval_ap(y_true=y_true, y_pred=y_pred)

        scheduler.step()

        # -------------------------
        # VALIDATION LOOP
        # -------------------------
        model.eval()
        real_val = []
        pred_val = []
        with torch.no_grad():
            for valdata in valoader:
                valdata = valdata.to(device)
                val_classify = model(valdata.x, valdata.edge_index, valdata.batch, device)

                val_loss = criterion(val_classify, valdata.y)
                real_val.append(valdata.y)
                pred_val.append(val_classify)

        val_y_true = torch.cat(real_val, dim=0)
        val_y_pred = torch.cat(pred_val, dim=0)
        val_perf = eval_ap(y_true=val_y_true, y_pred=val_y_pred)

        # Save checkpoint if the validation performance is improved
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            when = epoch
            # torch.save(model.state_dict(), checkpoint_path)

        # Print/log metrics
        print(
            f"Epoch: {epoch:03d}, "
            f"Loss: {loss.item():.4f}, "
            f"Train Acc: {train_perf:.4f}, "
            f"Val_Loss: {val_loss.item():.4f}, "
            f"Val Acc: {val_perf:.4f}"
        )

        wandb.log({"Train Acc": train_perf})
        wandb.log({"Val Acc": val_perf})
        wandb.log({"Train Loss": loss})
        wandb.log({"Val Loss": val_loss})
        wandb.log({"Epoch": epoch})

    # -------------------------
    # TESTING
    # -------------------------
    # Load the best checkpoint
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint)

    real_test = []
    pred_test = []
    model.eval()
    with torch.no_grad():
        for testdata in testloader:
            testdata = testdata.to(device)
            test_classify = model(testdata.x, testdata.edge_index, testdata.batch, device)

            real_test.append(testdata.y)
            pred_test.append(test_classify)

    y_test_true = torch.cat(real_test, dim=0)
    y_test_pred = torch.cat(pred_test, dim=0)
    test_perf = eval_ap(y_true=y_test_true, y_pred=y_test_pred)

    print(f"Best Validation Acc: {best_val_perf:.4f} (Epoch {when})")
    print(f"Test Acc: {test_perf:.4f}")
    wandb.log({"Test Acc": test_perf})
    wandb.finish()

    # Optionally return metrics, model, etc.
    return test_perf


# Replace the bottom portion of your script (everything after defining 'config')
# with something like this:

# Define a list of seeds and K values to test:

K_values = [5,6,7,8,9,10]
seeds = [0,42,100]
step_size=[0.05,0.1,0.2,0.5]
dissipative_force=0.1
hidden=300
num_layers=[3,4]
mlp_layers=3

for k_val in K_values:
    for num_layer in num_layers:
        for step in step_size:
            for seed_val in seeds:
                # Update config for each run
                config = dict(
                    hidden_dim=hidden,
                    K_hops=k_val,
                    batch_size=batch_size,
                    num_layers=num_layer,
                    mlp_layers=mlp_layers,
                    learning_rate=lr,
                    seed=seed_val
                )
                
                # Each loop run initializes a new W&B run for clarity
                wandb.init(
                    project="PeptideFunc_Euler",
                    name=f"Cheb_K{k_val}_seed{seed_val}",
                    config=config
                )
                
                # Rebuild your model, optimizer, etc. here:
                model = EulerModel(hidden, k_val, num_layer, mlp_layers, num_classes,step,dissipative_force).to(device) #model = EulerModel(args.hidden,args.K,args.num_layers,args.mlp_layers,num_classes,args.step_size,args.dissipative_force).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                criterion = torch.nn.CrossEntropyLoss()
                
                # Run the pipeline for this (K, seed) combo
                test_perf = run_entire_pipeline(
                    model,
                    seed_val,
                    trainloader,
                    valoader,
                    testloader,
                    device,
                    optimizer,
                    scheduler,
                    criterion,
                    eval_ap
                )

                # Close the W&B run after the experiment
                wandb.finish()
