export NODES=1
export GPUS=1
export MASTER_ADDR=127.0.0.1
export PORT=19503
export CUDA_VISIBLE_DEVICES=0
export PRETRAIN_CHKPT="YourCheckpointPath"

OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node $GPUS --master_addr $MASTER_ADDR --master_port $PORT \
    main.py \
    --config cfgs/Pseudo3D_ScanObjNN/SWI_transformer_base_objectbg.yaml \
    --exp_name P3P_sparse_base_finetune_objectbg \
    --val_freq 1 \
    --sync_bn \
    --num_workers 6 \
    --finetune_model \
    --ckpts ${PRETRAIN_CHKPT}