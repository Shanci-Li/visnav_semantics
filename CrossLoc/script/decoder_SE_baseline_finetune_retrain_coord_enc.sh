#!/bin/bash
#SBATCH --chdir /home/qyan/TransPose
#SBATCH --nodes 1
#SBATCH --cpus-per-task 20
#SBATCH --ntasks 1
#SBATCH --account topo
#SBATCH --mem 96G
#SBATCH --time 72:00:00
#SBATCH --partition gpu
#SBATCH --reservation topo
#SBATCH --qos gpu
#SBATCH --gres gpu:1

LR=1e-4
EPOCHS=150
SCR_TOL=50.0

BATCH_SIZE=8
CKPT_DIR=/scratch/izar/$USER/ckpt-transpose

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PROJ_DIR="$(dirname $SCRIPT_DIR)"
cd $PROJ_DIR
PROJ_DIR=$(pwd)

source /home/qyan/venvtranspose/bin/activate


DATASET=$1
if [ -z "$DATASET" ]
then
  echo "DATASET is empty"
  DATASET="urbanscape"
else
  echo "DATASET is set"
fi
echo $DATASET


TASK=$2
if [ -z "$TASK" ]
then
  echo "TASK is empty"
  TASK=NONE
else
  echo "TASK is set"
fi
echo $TASK


NET_DEPTH=$3
if [ -z "$NET_DEPTH" ]
then
  echo "NET_DEPTH is empty"
  NET_DEPTH=FULL
else
  echo "NET_DEPTH is set"
fi
echo $NET_DEPTH


# from this pretrained model
PT_SIM_DATA_CHUNK=$4
if [ -z "$PT_SIM_DATA_CHUNK" ]
then
  echo "PT_SIM_DATA_CHUNK is empty"
  PT_SIM_DATA_CHUNK=1.0
else
  echo "PT_SIM_DATA_CHUNK is set"
fi
echo $PT_SIM_DATA_CHUNK


PT_REAL_DATA_DOM=$5
if [ -z "$PT_REAL_DATA_DOM" ]
then
  echo "PT_REAL_DATA_DOM is empty"
  PT_REAL_DATA_DOM="in_place"
else
  echo "PT_REAL_DATA_DOM is set"
fi
echo "$PT_REAL_DATA_DOM"


PT_REAL_DATA_CHUNK=$6
if [ -z "$PT_REAL_DATA_CHUNK" ]
then
  echo "PT_REAL_DATA_CHUNK is empty"
  PT_REAL_DATA_CHUNK=1.0
else
  echo "PT_REAL_DATA_CHUNK is set"
fi
echo $PT_REAL_DATA_CHUNK


# to use this amount of training data
SIM_DATA_CHUNK=$7
if [ -z "$SIM_DATA_CHUNK" ]
then
  echo "SIM_DATA_CHUNK is empty"
  SIM_DATA_CHUNK=1.0
else
  echo "SIM_DATA_CHUNK is set"
fi
echo $SIM_DATA_CHUNK


REAL_DATA_DOM=$8
if [ -z "$REAL_DATA_DOM" ]
then
  echo "REAL_DATA_DOM is empty"
  REAL_DATA_DOM="in_place"
else
  echo "REAL_DATA_DOM is set"
fi
echo "$REAL_DATA_DOM"


REAL_DATA_CHUNK=$9
if [ -z "$REAL_DATA_CHUNK" ]
then
  echo "REAL_DATA_CHUNK is empty"
  REAL_DATA_CHUNK=1.0
else
  echo "REAL_DATA_CHUNK is set"
fi
echo $REAL_DATA_CHUNK


UNC=${10}
if [ -z "$UNC" ]
then
  echo "UNC is empty"
  UNC="none"
else
  echo "UNC is set"
fi
echo $UNC


CUDA_ID=${11}
if [ -z "$CUDA_ID" ]
then
  echo "CUDA_ID is empty"
  CUDA_ID=0
else
  echo "CUDA_ID is set"
fi
echo $CUDA_ID
export CUDA_VISIBLE_DEVICES=${CUDA_ID}


SIM_DATA_CHUNK=$(printf "%.2f" ${SIM_DATA_CHUNK})
REAL_DATA_CHUNK=$(printf "%.2f" ${REAL_DATA_CHUNK})
if [ "$SIM_DATA_CHUNK" == "0.00" ];
then
    if [ $TASK == "coord" ]
    then
      echo "Use more epochs for sim-to-real paired data tuning: 150 -> 1001"
      EPOCHS=1001
    fi
fi


PT_SIM_DATA_CHUNK=$(printf "%.2f" ${PT_SIM_DATA_CHUNK})
PT_REAL_DATA_CHUNK=$(printf "%.2f" ${PT_REAL_DATA_CHUNK})

ENC_FT_COORD=$PROJ_DIR/weights/encoders-finetuned/${DATASET}/${PT_REAL_DATA_DOM}/coord/model-sc${PT_SIM_DATA_CHUNK}-rc${PT_REAL_DATA_CHUNK}.net
ENC_FT_DEPTH=$PROJ_DIR/weights/encoders-finetuned/${DATASET}/${PT_REAL_DATA_DOM}/depth/model-sc${PT_SIM_DATA_CHUNK}-rc${PT_REAL_DATA_CHUNK}.net
ENC_FT_NORMAL=$PROJ_DIR/weights/encoders-finetuned/${DATASET}/${PT_REAL_DATA_DOM}/normal/model-sc${PT_SIM_DATA_CHUNK}-rc${PT_REAL_DATA_CHUNK}.net
ENC_FT_SEMANTICS=$PROJ_DIR/weights/encoders-finetuned/${DATASET}/${PT_REAL_DATA_DOM}/semantics/model-sc${PT_SIM_DATA_CHUNK}-rc${PT_REAL_DATA_CHUNK}.net

SP_SESSION=''
if [ "$PT_REAL_DATA_DOM" == "in_place" ];
  then
    SP_SESSION='load-pt'$PT_SIM_DATA_CHUNK-ip-ft${PT_REAL_DATA_CHUNK}'-lls'
  else
    SP_SESSION='load-pt'$PT_SIM_DATA_CHUNK-oop-ft${PT_REAL_DATA_CHUNK}'-lls'
fi

# zero-shot learning, no pairwise sim-to-real data is used
if [ "$PT_REAL_DATA_CHUNK" == "0.00" ] && [ "$REAL_DATA_CHUNK" == "0.00" ];
  then
    ENC_FT_COORD=$PROJ_DIR/weights/encoders-pretrained/${DATASET}/coord/model-sc-${PT_SIM_DATA_CHUNK}.net
    ENC_FT_DEPTH=$PROJ_DIR/weights/encoders-pretrained/${DATASET}/depth/model-sc-${PT_SIM_DATA_CHUNK}.net
    ENC_FT_NORMAL=$PROJ_DIR/weights/encoders-pretrained/${DATASET}/normal/model-sc-${PT_SIM_DATA_CHUNK}.net
    ENC_FT_SEMANTICS=$PROJ_DIR/weights/encoders-pretrained/${DATASET}/semantics/model-sc-${PT_SIM_DATA_CHUNK}.net
fi

echo start at `date`


case $TASK in
  "coord")
    case $NET_DEPTH in
      "TINY")
        python3 finetune_decoder_single_task.py ${DATASET} --task ${TASK} \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --softclamp 100 --hardclamp 1000 \
          --uncertainty ${UNC} --auto_resume --tiny --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk ${SIM_DATA_CHUNK} \
          --coord_weight ${ENC_FT_COORD}  --depth_weight ${ENC_FT_DEPTH}  --normal_weight ${ENC_FT_NORMAL}  --semantics_weight ${ENC_FT_SEMANTICS} \
          --encoders coord semantics --session "${SP_SESSION}" \
          --reuse_coord_encoder --unfreeze_coord_encoder --no_lr_scheduling
        ;;
      "FULL")
        python3 finetune_decoder_single_task.py ${DATASET} --task ${TASK} \
          --learningrate ${LR} --epochs ${EPOCHS} --inittolerance ${SCR_TOL}  --batch_size ${BATCH_SIZE} \
          --softclamp 100 --hardclamp 1000 \
          --uncertainty ${UNC} --auto_resume --ckpt_dir ${CKPT_DIR} \
          --real_data_domain ${REAL_DATA_DOM} --real_data_chunk ${REAL_DATA_CHUNK} --sim_data_chunk ${SIM_DATA_CHUNK} \
          --coord_weight ${ENC_FT_COORD}  --depth_weight ${ENC_FT_DEPTH}  --normal_weight ${ENC_FT_NORMAL}  --semantics_weight ${ENC_FT_SEMANTICS} \
          --encoders coord semantics --session "${SP_SESSION}" \
          --reuse_coord_encoder --unfreeze_coord_encoder --no_lr_scheduling
        ;;
      *)
        echo "$NET_DEPTH is not a pre-specified NET_DEPTH, do nothing..."
    esac
    ;;

  *)
    echo "$TASK is not a pre-specified task, do nothing..."
esac

echo finished at `date`
