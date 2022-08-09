set -x

# --output_dir ./vox_result/face_to_face_trainset_l1_loss \
# --output_dir ./vox_result/face_to_face_demoset_l1_loss \

########## For cross identity ##########
# python inference_face_to_face.py \
# --config ./config/face_to_face.yaml \
# --name face_to_face_vox_l1_loss_new_mask \
# --no_resume \
# --output_dir ./vox_result/trainset_fixed_reference_image_another_exp_erode_11 \
# --single_gpu

########## For HDTF dataset inference ##########
python inference_face_to_face.py \
--config ./config/face_to_face_HDTF.yaml \
--name HDTF_face_to_face \
--no_resume \
--output_dir ./HDTF_result/trainset_fixed_ref_img_cross_id \
--single_gpu