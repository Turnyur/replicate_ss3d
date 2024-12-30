#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00


module load cuda/12.3.0 
conda activate dev_ss3d



srun python3 -c 'import torch; print(torch.cuda.is_available());print()'


# eval $(/home/hpc/iwi9/iwi9117h/mambaforge/bin/conda shell.bash hook)"

#python synth_pretraining.py resources.gpus=1 resources.num_nodes=1 resources.use_cluster=True 
#logging.name=synthetic_pretraining optim.use_scheduler=True



#salloc.tinygpu --gres=gpu:1 --time=00:10:00 && conda activate dev_ss3d && python3 -c 'import torch; print(torch.cuda.is_available());print()'