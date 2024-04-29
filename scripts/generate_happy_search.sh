cd ..
emo_data=("happiness" "happiness")
attribute_list=("happy2sad" "sad2happy")

for ((i = 0; i < 2; i++));
do
    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-1k-search-3iter \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 10 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 3 \
        --return_hiddens \
        --max_norm 100 \
        --add_everything \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-search-3iter \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 1 \
        --search \
        --do_sample \
        --batchsize 4 \
        --init_coeff -2.5 \
        --iteration 3 \
        --return_hiddens \
        --max_norm 100 \
        --add_everything \
        --add_inst \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

done