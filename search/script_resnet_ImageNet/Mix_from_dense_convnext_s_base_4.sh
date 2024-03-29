#/bin/bash

CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.7 \
--config ./configs/config_convnext_small_img_mix_from_dense.yaml

CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 25087 \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.65 \
--config ./configs/config_convnext_small_img_mix_from_dense.yaml \
--port 25087


# FLOPs
CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 25011 \
../find_mix_from_dense_imagenet.py \
--target_sparse_flops_ratio 0.5 \
--config ./configs/config_convnext_small_img_mix_from_dense.yaml \
--port 25011

CUDA_VISIBLE_DEVICES=1 \
python -m torch.distributed.launch --nproc_per_node=1 --master_port 25087 \
../find_mix_from_dense_imagenet.py \
--target_sparse_flops_ratio 0.5 \
--config ./configs/config_convnext_small_img_mix_from_dense.yaml \
--port 25087

