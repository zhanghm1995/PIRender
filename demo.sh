set -x


# python demo.py \
# --config ./config/face_to_face_HDTF.yaml \
# --name HDTF_face_to_face_no_adain \
# --no_resume \
# --output_dir ./HDTF_AAAI_result/demo_audio_kanghui_PPE \
# --single_gpu

# --name HDTF_face_to_face_no_adain_mask_aug_AS \

python demo.py \
--config ./config/face_to_face_HDTF.yaml \
--name ablation_HDTF_face_to_face_wo_blended_img \
--no_resume \
--output_dir ./HDTF_AAAI_result/ablation_wo_face_blend/paper_PE_val_condition_WDA_KimSchrier_000_001_audio_WRA_KellyAyotte_000 \
--single_gpu