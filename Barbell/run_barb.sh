#!/bin/bash
#SBATCH --job-name barb # Name for your job
#SBATCH --ntasks 4              # Number of (cpu) tasks
#SBATCH --time  400         # Runtime in minutes.
#SBATCH --mem 24000             # Reserve x GB RAM for the job
#SBATCH --partition gpu         # Partition to submit
#SBATCH --qos staff             # QOS
#SBATCH --gres gpu:a100:1            # Reserve 1 GPU for usage (titanrtx, gtx1080,a100)
#SBATCH --chdir ...Your directory...

# RUN BENCHMARK
eval "$(conda shell.bash hook)"
conda activate grop

# python -m run --yaml_name barbell-cheb.yml
# python -m run --yaml_name barbell-eulercheb.yml
# python -m run_multiple --yaml_name barbell-cheb.yml

# python -m run_multiple --yaml_name barbell-eulercheb.yml
python -m run_multiple --yaml_name barbell-cheb.yml

# python -m run_multiple --yaml_name barbell-nondischeb.yml
# python -m run_multiple --yaml_name barbell-nondischebv2.yml
