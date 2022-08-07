set -x

# --output_dir ./vox_result/face_to_face_trainset_l1_loss \
# --output_dir ./vox_result/face_to_face_demoset_l1_loss \

########## For cross identity ##########
python inference_face_to_face.py \
--config ./config/face_to_face.yaml \
--name face_to_face_vox_l1_loss \
--no_resume \
--output_dir ./vox_result/debug \
--single_gpu