# StableCheb

A unified repository of Chebyshev-based spectral graph neural network implementations for a variety of tasks, including benchmarks on the Open Graph Benchmark (OGB), graph property prediction (GraphProp), peptide modeling, and the Barbell clique model.

---

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [OGB Benchmarks](#ogb-benchmarks)
  - [Graph Property Prediction (GraphProp)](#graph-property-prediction-graphprop)
  - [Peptide Modeling](#peptide-modeling)
  - [Barbell Clique Model (Barbell)](#barbell-clique-model-barbell)
  - [Jacobian Analysis (jacobian_cheb.py)](#jacobian-analysis-jacobian_chebpy)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Stable-ChebNet**: A stability-enhanced Chebyshev spectral network.
- **ChebNet, GCN, GAT** variants for GraphProp tasks.
- **OGB Datasets**: Out-of-the-box scripts for `ogb-arxiv` and `ogb-proteins`, with custom Euler-ChebNet support.
- **Peptide Tasks**: Structure and function prediction using Chebyshev networks.
- **Barbell Model**: Clique-based toy model for spectral analysis.

---

## Dependencies

- Python 3.7+
- [PyTorch](https://pytorch.org/) (1.8+ recommended)
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [OGB](https://github.com/snap-stanford/ogb)

Install via pip:

```bash
pip install torch-geometric ogb
```

Additional dependencies may be specified in each subfolder’s `requirements.txt` or `environment.yml`.

---

## Installation

1. Clone this repository:

   ```bash
git clone https://github.com/your-username/StableCheb.git
cd StableCheb
```

2. (Optional) Create and activate a conda environment:

   ```bash
conda create -n stablecheb python=3.8
conda activate stablecheb
```

3. Install the core dependencies:

   ```bash
pip install -r requirements.txt
# or manually:
pip install torch-geometric ogb
```

4. (Optional) Install extras per subproject:

   ```bash
# For GraphProp
cd GraphProp
# adjust your conda env in run_models.sh (line 13)
cd ..
```

---

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

### Jacobian Analysis (`jacobian_cheb.py`)

Compute and visualize the Jacobian of a Chebyshev convolution layer:

```bash
python jacobian_cheb.py [--help]
```

Refer to the docstring at the top of the file for available flags and usage.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add YourFeature"`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please follow the coding style in the existing modules and add tests where appropriate.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
