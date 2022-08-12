set -x

ZHM_PATH="/221019051/"
YZH_PATH="/221019041/"

PYTHON=python
if [ -d "$ZHM_PATH" -o -d "$YZH_PATH" ]; then
    echo "In AIStation platform"
    PYTHON=/root/miniconda3/envs/py36-torch100-cu11/bin/python
    if [ -f "$PYTHON" ]; then
        PYTHON=/root/miniconda3/envs/py36-torch100-cu11/bin/python
    else
        PYTHON=python
    fi
    cd /221019051/Research/Face/PIRender
fi

# ${PYTHON} -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 train.py \
# --config ./config/face_ours.yaml \
# --name face_vox


## For face to face training
# ${PYTHON} -m torch.distributed.launch --nproc_per_node=1 --master_port 22222 train.py \
# --config ./config/face_to_face.yaml \
# --name face_to_face_vox \
# --single_gpu


# ${PYTHON} train.py \
# --config ./config/face_to_face.yaml \
# --name face_to_face_vox_l1_loss_new_mask \
# --single_gpu

## For training HDTF #####
${PYTHON} train.py \
--config ./config/face_to_face_HDTF.yaml \
--name HDTF_face_to_face_no_adain_mask_augment \
--single_gpu