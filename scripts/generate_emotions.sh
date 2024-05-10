cd ..
# emo_data=("anger" "fear" "surprise" "disgust")
# attribute_list=("angry2peaceful" "afraid2fearless" "surprised2calm" "disgusted2satisfied")
emo_data=("fear")
attribute_list=("afraid2fearless")

for ((i = 0; i < 4; i++));
do
    # CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
    #     --attribute ${attribute_list[$i]} \
    #     --output_name ${attribute_list[$i]}-final \
    #     --start_from_idx 0 \
    #     --max_num_data 100 \
    #     --epoch 20 \
    #     --search \
    #     --do_sample \
    #     --batchsize 4 \
    #     --init_coeff -2.5 \
    #     --iteration 2 \
    #     --return_hiddens \
    #     --add_prefix \
    #     --add_inst \
    #     --max_norm 0.2 \
    #     --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=1 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 5 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --add_inst \
        --max_norm 0.2 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json
done