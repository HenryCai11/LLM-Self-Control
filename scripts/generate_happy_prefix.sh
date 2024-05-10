cd ..
emo_data=("happiness")
attribute_list=("happy2sad")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-2iter-05norm \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 20 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -3 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --add_inst \
        --max_norm 0.5 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-2iter-05norm \
        --start_from_idx 100 \
        --max_num_data 103 \
        --epoch 2 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -3 \
        --iteration 2 \
        --return_hiddens \
        --add_inst \
        --add_prefix \
        --max_norm 0.5 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

done