cd ..
attribute_list=("toxic2nontoxic" "nontoxic2toxic")

for ((i = 0; i < 2; i++));
do
    CUDA_VISIBLE_DEVICES=2 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-2k-search \
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

    CUDA_VISIBLE_DEVICES=2 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-search \
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