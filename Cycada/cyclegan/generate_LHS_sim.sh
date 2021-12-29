#!/bin/bash
#SBATCH --chdir /home/shanli/visnav_semantics/cycada/cyclegan
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 32G
#SBATCH --time 5:59:59
#SBATCH --partition gpu
#SBATCH --qos gpu
#SBATCH --gres gpu:1

cd /home/shanli/visnav_semantics/Cycada/cyclegan
source /home/shanli/visnav_semantics/baseline/venvs/seg/bin/activate
module load gcc cmake python
export PYTHONPATH=/home/shanli/visnav_semantics:$PYTHONPATH

python test.py --name comballaz_pairwise --which_epoch 20 --model cycle_gan_semantic --dataroot /work/topo/VNAV/remade-data-fullsize/comballaz --phase train_sim train_drone_real
cp -r /work/topo/VNAV/remade-data-fullsize/comballaz/train_sim/styled_as_target/comballaz_pairwise/train_drone_sim_20/images/fake_B/* /work/topo/VNAV/remade-data-fullsize/comballaz/train_sim/styled_as_target
rm -rf /work/topo/VNAV/remade-data-fullsize/comballaz/train_sim/styled_as_target/comballaz_pairwise