cd ..
CUDA_VISIBLE_DEVICES=0

# truthful_attributes=("truthful" "notruthful")
# for attribute in "${truthful_attributes[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
#         --attribute $attribute \
#         --output_name "$attribute-1k" \
#         --start_from_idx 100 \
#         --max_num_data 300 \
#         --epoch 5 \
#         --search \
#         --do_sample \
#         --batchsize 4 \
#         --init_coeff -2.5 \
#         --iteration 2 \
#         --return_hiddens \
#         --max_norm 100

#     CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
#         --attribute $attribute \
#         --output_name "$attribute-eval" \
#         --start_from_idx 0 \
#         --max_num_data 100 \
#         --epoch 1 \
#         --search \
#         --do_sample \
#         --batchsize 4 \
#         --init_coeff -2.5 \
#         --iteration 2 \
#         --return_hiddens \
#         --max_norm 100
# done


# emo_data=("anger" "anger" "fear" "fear" "surprise" "surprise" "disgust" "disgust")
# attribute_list=("angry2peaceful" "peaceful2angry" "afraid2fearless" "fearless2afraid" "surprised2calm" "calm2surprised" "disgusted2satisfied" "satisfied2disgusted")
emo_data=("anger" "fear" "fear" "surprise" "surprise" "disgust" "disgust")
attribute_list=("peaceful2angry" "afraid2fearless" "fearless2afraid" "surprised2calm" "calm2surprised" "disgusted2satisfied" "satisfied2disgusted")

for ((i = 0; i < 8; i++));
do
    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-1k \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 15 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 100 \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-eval \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 1 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 100 \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json
done