#!/bin/bash -l

#SBATCH --gres=gpu:2

#SBATCH --time=12:50:00





set -x
set -v

module load python
module load cuda/11.8.0
conda activate pytorch_ss3d


srun python3 -c 'import torch; print(torch.cuda.is_available())'

#srun python synth_pretraining.py resources.gpus=0 resources.num_nodes=1 resources.use_cluster=False logging.name=synthetic_pretraining optim.use_scheduler=False

python synth_pretraining.py resources.gpus=2 data.bs_train=2 resources.num_nodes=1 resources.use_cluster=False logging.name=airplane_tester optim.use_scheduler=True

# Todo 

touch ready #SBATCH --partition=a100


