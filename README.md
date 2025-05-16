# StableCheb

---

## Dependencies

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://github.com/snap-stanford/ogb)

## Project Structure

```
StableCheb/
├── Barbell/            # Clique-based Barbell model experiments
├── GraphProp/          # Graph property prediction scripts & configs
├── OGB/                # Scripts for OGB datasets (arxiv, proteins)
├── Peptides/           # Peptide function & structure prediction
├── jacobian_cheb.py    # Jacobian analysis for Chebyshev conv
└── README.md           # This document
```

---

## Usage

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
  python ogbproteins_euler.py     --model_name eulerchebnet     --hidden_channels 100     --num_layers 2     --K 4     --lr 0.0005
  ```

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

### Peptide Modeling

Two pipelines are provided under `Peptides/Stable/`:

- **Function Prediction**:
  ```bash
  cd Peptides/Stable
  python ChebStable_peptide.py
  ```

- **Structure Prediction**:
  ```bash
  cd Peptides/Stable
  python ChebStable_Struct.py
  ```

Adjust hyperparameters in the scripts or configuration files as needed.

### Barbell Clique Model (Barbell)

Launch the Barbell experiments via Slurm:

```bash
cd Barbell
sbatch run_barb.sh
```

Specify clique sizes and other parameters in the `configs/*.yaml` files.


This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
