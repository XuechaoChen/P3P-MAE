export NODES=1
export GPUS=4
export RANK=0
export MASTER_ADDR=127.0.0.1
export PORT=19501

torchrun --nnodes=$NODES --nproc_per_node=$GPUS --node_rank=$RANK --master_addr=$MASTER_ADDR --master-port=$PORT\
    main.py \
    --launcher pytorch \
    --sync_bn \
    --config cfgs/Pseudo3D_pretrain/P3P_sparse_small.yaml \
    --exp_name P3P_sparse_small \
    --val_freq 1 \
    --num_workers 16