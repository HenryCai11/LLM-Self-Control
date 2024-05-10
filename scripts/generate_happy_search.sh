cd ..
emo_data=("happiness")
attribute_list=("happy2sad")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-1k-search-binary \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 10 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -3 \
        --iteration 1 \
        --return_hiddens \
        --max_norm 100 \
        --add_everything \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-search-binary \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 1 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -3 \
        --iteration 1 \
        --return_hiddens \
        --max_norm 100 \
        --add_everything \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

done