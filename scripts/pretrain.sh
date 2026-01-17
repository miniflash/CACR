GPU_ID=0

DATASET='s3dis'
SPLIT=0
DATA_PATH='/root/autodl-fs/S3DIS/blocks_bs1_s1'
SAVE_PATH='./exps/log_s3dis/'

NUM_POINTS=2048
PC_ATTRIBS='xyz'

EVAL_INTERVAL=10
BATCH_SIZE=32
NUM_WORKERS=32
NUM_EPOCHS=150
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5

args=(--phase 'pretrain' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --n_iters $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --pretrain_lr $LR --pretrain_weight_decay $WEIGHT_DECAY
      --pretrain_step_size $DECAY_STEP --pretrain_gamma $DECAY_RATIO)

CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"
