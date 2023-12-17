#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=torch
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=hs4543@nyu.edu

module purge
module load cuda/11.1.74


singularity exec --nv  --overlay /scratch/hs4543/CVProj/my_pytorch.ext3:ro  /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash -c "source /ext3/env.sh; python runtime.py --inference --pre_trained_disc ./checkpoints/disc_190.pth --pre_trained_gen ./checkpoints/gen_190.pth"