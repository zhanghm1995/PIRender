set -x

# python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference.py \
# --config ./config/face_demo.yaml \
# --name face \
# --no_resume \
# --output_dir ./vox_result/face_reenactment

########## For cross identity ##########
python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 inference.py \
--config ./config/face_demo.yaml \
--name face \
--no_resume \
--output_dir ./vox_result/face_reenactment_cross \
--cross_id