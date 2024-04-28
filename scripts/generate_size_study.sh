cd ..
# emo_data=("anger" "fear" "fear" "surprise" "surprise" "disgust" "disgust")
# attribute_list=("peaceful2angry" "afraid2fearless" "fearless2afraid" "surprised2calm" "calm2surprised" "disgusted2satisfied" "satisfied2disgusted")

attribute_list=("toxic2nontoxic" "nontoxic2toxic")

for ((i = 0; i < 2; i++));
do
    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-2k \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 20 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 100 \
        --add_everything \

    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
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
        --add_everything \

done



# emo_data=("happiness" "happiness")
# attribute_list=("happy2sad" "sad2happy")

# for ((i = 0; i < 2; i++));
# do
#     CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds \
#         --attribute ${attribute_list[$i]}\
#         --output_name ${attribute_list[$i]}-2k \
#         --start_from_idx 0 \
#         --max_num_data 100 \
#         --epoch 20 \
#         --search \
#         --do_sample \
#         --batchsize 4 \
#         --init_coeff -2.5 \
#         --iteration 2 \
#         --return_hiddens \
#         --max_norm 100 \
#         --add_inst \
#         --add_everything \
#         --data_path benchmarks/emotions/${emo_data[$i]}.json

# done

# CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds \
#     --attribute happy2sad \
#     --output_name happy2sad-eval \
#     --start_from_idx 100 \
#     --max_num_data 100 \
#     --epoch 1 \
#     --search \
#     --do_sample \
#     --batchsize 4 \
#     --init_coeff -2.5 \
#     --iteration 2 \
#     --return_hiddens \
#     --max_norm 100 \
#     --add_inst \
#     --add_everything \
#     --data_path benchmarks/emotions/happiness.json

# CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds \
#     --attribute sad2happy \
#     --output_name sad2happy-eval \
#     --start_from_idx 100 \
#     --max_num_data 100 \
#     --epoch 1 \
#     --search \
#     --do_sample \
#     --batchsize 4 \
#     --init_coeff -2.5 \
#     --iteration 2 \
#     --return_hiddens \
#     --max_norm 100 \
#     --add_inst \
#     --add_everything \
#     --data_path benchmarks/emotions/happiness.json