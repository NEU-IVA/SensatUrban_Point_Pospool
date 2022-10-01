python -m torch.distributed.launch --nproc_per_node 8 train_sensat.py --batch_size 2 --cfg ./configs/pospool.yaml > ./running_log &
