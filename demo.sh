set -x


# python demo.py \
# --config ./config/face_to_face_HDTF.yaml \
# --name HDTF_face_to_face_no_adain \
# --no_resume \
# --output_dir ./HDTF_AAAI_result/demo_audio_kanghui_PPE \
# --single_gpu


python demo.py \
--config ./config/face_to_face_HDTF.yaml \
--name HDTF_face_to_face_no_adain_mask_aug_AS \
--no_resume \
--output_dir ./HDTF_AAAI_result/demo_debug \
--single_gpu