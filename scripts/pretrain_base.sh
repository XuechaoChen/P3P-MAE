export NODES=1
export GPUS=4
export MASTER_ADDR=127.0.0.1
export PORT=19501

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node $GPUS --master_addr $MASTER_ADDR --master_port $PORT \
    main.py \
    --launcher pytorch \
    --sync_bn \
    --config cfgs/Pseudo3D_pretrain/P3P_sparse_base.yaml \
    --exp_name P3P_sparse_base \
    --val_freq 1 \
    --num_workers 16