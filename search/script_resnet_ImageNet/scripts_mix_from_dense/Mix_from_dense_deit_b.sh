#/bin/bash

python -m torch.distributed.launch --nproc_per_node=1  \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.625 \
--config ./configs/config_deit_base_img_mix_from_dense.yaml


python -m torch.distributed.launch --nproc_per_node=1  \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.65  \
--config ./configs/config_deit_base_img_mix_from_dense.yaml

python -m torch.distributed.launch --nproc_per_node=1  \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.675 \
--config ./configs/config_deit_base_img_mix_from_dense.yaml

python -m torch.distributed.launch --nproc_per_node=1  \
../find_mix_from_dense_imagenet.py \
--target_sparsity 0.7 \
--config ./configs/config_deit_base_img_mix_from_dense.yaml

