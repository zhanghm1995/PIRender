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
    cd /221019051/Research/Face/hififace
fi

${PYTHON} -m torch.distributed.launch --nproc_per_node=4 --master_port 12345 train.py \
--config ./config/face_ours.yaml \
--name face_vox