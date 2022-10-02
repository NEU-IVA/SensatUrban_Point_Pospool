# CUDA_VISIBLE_DEVICES=$1
python -m torch.distributed.launch \
	--nproc_per_node $1 train_sensat.py \
	--batch_size 3 \
	--cfg ./configs/pospool.yaml \
	--data_root "/mnt/SSD/zhzhang/dataset_500" \
#  	> ./running_log &
