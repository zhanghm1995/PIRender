set -x

python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 train.py \
--config ./config/face.yaml \
--name face_vox