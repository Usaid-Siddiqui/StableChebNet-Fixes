#!/bin/bash
#SBATCH --job-name euler # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  4200         # Runtime in minutes.
#SBATCH --mem 24000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:titanrtx:1            # Reserve 1 GPU for usage (titanrtx, gtx1080)
#SBATCH --chdir ... your_path_here.../

# RUN BENCHMARK
eval "$(conda shell.bash hook)"
conda activate ... your_env_name...

python -u main.py --data_name GraphProp --task ecc --model_name Euler_GraphProp 
python -u main.py --data_name GraphProp --task diam --model_name Euler_GraphProp
python -u main.py --data_name GraphProp --task dist --model_name Euler_GraphProp