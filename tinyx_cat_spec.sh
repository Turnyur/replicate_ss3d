#!/bin/bash -l

#SBATCH --gres=gpu:1
#SBATCH --time=23:30:00


set -x
set -v


module load cuda/11.8.0
conda activate pytorch_ss3d


srun python3 -c 'import torch; print(torch.cuda.is_available())'

#srun python synth_pretraining.py resources.gpus=0 resources.num_nodes=1 resources.use_cluster=False logging.name=synthetic_pretraining optim.use_scheduler=False


#srun python cat_spec_training.py resources.use_cluster=False resources.gpus=8 resources.num_nodes=1 logging.name=catspecific_training render.cam_num=10 render.num_pre_rend_masks=10 data=generic_img_mask data.bs_train=2 data.train_dataset_file=/home/hpc/iwi9/iwi9117h/dev/stage2/CUB_200_2011/train_data.csv data.val_dataset_file=/home/hpc/iwi9/iwi9117h/dev/stage2/CUB_200_2011/val_data.csv data.class_ids=bird_class optim.stage_one_epochs=10 optim.max_epochs=50 optim.lr=0.00001 optim.use_pretrain=True optim.checkpoint_path=/home/hpc/iwi9/iwi9117h/dev/job_outputs/synthetic_pretraining/version_1/checkpoints/checkpoint_epoch=474.ckpt



python cat_spec_training.py resources.gpus=1 logging.name=catspecific_training render.cam_num=10 render.num_pre_rend_masks=10 data=generic_img_mask optim.stage_one_epochs=10 optim.max_epochs=50 optim.lr=0.00001 data.bs_train=1


touch ready


