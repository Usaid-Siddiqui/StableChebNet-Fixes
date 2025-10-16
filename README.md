## 📘 Paper: *Return of ChebNet — Understanding and Improving an Overlooked GNN on Long-Range Tasks*
[**arXiv:2506.07624**](https://arxiv.org/abs/2506.07624)

**Authors:** Ali Hariri, Álvaro Arroyo, Alessio Gravina, Moshe Eliasof, Carola-Bibiane Schönlieb, Davide Bacciu, Kamyar Azizzadenesheli, Xiaowen Dong, and Pierre Vandergheynst

![Alt text](assets/overview.png.png)
---

## Dependencies

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://github.com/snap-stanford/ogb)

## Usage


### Graph Property Prediction (GraphProp)

1. Edit your conda environment in `GraphProp/graph_prop_pred/run_models.sh` (see line 13).
2. Run the shell script to launch experiments:

   ```bash
   cd GraphProp/graph_prop_pred
   bash run_models.sh
   ```

3. Or invoke `main.py` directly:

   ```bash
   python -u main.py      --data_name GraphProp      --task ecc      --model_name Euler_GraphProp
   ```

Supported `--model_name` values:

- `Euler_GraphProp` (Stable-ChebNet)
- `Cheb_GraphProp` (ChebNet)
- `GCN_GraphProp` (Graph Convolutional Network)
- `GAT_GraphProp` (Graph Attention Network)
  
### Barbell Graphs task

Launch the Barbell experiments via Slurm:

```bash
cd Barbell
sbatch run_barb.sh
```

Specify clique sizes and other parameters in the `configs/*.yaml` files.

### OGB Benchmarks

Navigate into the `OGB` folder for dataset scripts:

- **arXiv citation network**:
  ```bash
  python ogb_datasets.py
  ```

- **Protein–protein interaction (OGB-Proteins)**:
  ```bash
  python ogbproteins.py
  ```

- **OGB-Proteins with Stable-ChebNet**:
  ```bash
  python ogbproteins_euler.py     --model_name eulerchebnet     --hidden_channels #insert_dim     --num_layers #insert_layers     --K #insert_Khops     --lr #insert_lr
  ```

### Peptides dataset

Two pipelines are provided under `Peptides/Stable/`:

- **Function Prediction**:
  ```bash
  cd Peptides/Stable
  python ChebStable_peptide.py
  ```

- **Structure Prediction**:
  ```bash
  cd Peptides/Stable
  python ChebStable_Struc.py
  ```

Adjust hyperparameters in the scripts or configuration files as needed.


