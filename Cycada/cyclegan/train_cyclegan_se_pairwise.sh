#!/bin/bash
#SBATCH --chdir /home/shanli/visnav_semantics/cycada/cyclegan
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 32G
#SBATCH --time 23:59:59
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1

cd /home/shanli/visnav_semantics/cycada/cyclegan
source /home/shanli/visnav_semantics/baseline/venvs/seg/bin/activate
module load gcc cmake python
export PYTHONPATH=/home/shanli/visnav_semantics:$PYTHONPATH

python -m visdom.server & python train.py --name cyclegan_se_pairwise --phase train_drone_sim train_drone_real --model cycle_gan_semantic


