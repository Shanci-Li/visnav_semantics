#!/bin/bash
#SBATCH --chdir /home/shanli/baseline
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 32G
#SBATCH --time 23:59:59
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1

cd /home/shanli/baseline
source venvs/seg/bin/activate
module load gcc cmake python

python train.py --loss DiceLoss
