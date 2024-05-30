cd ..
attribute_list=("helpfulharmless")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-final-gogogo \
        --start_from_idx 100 \
        --max_num_data 100 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --do_sample \
        --n_branches 3 \
        --binary \
        --batchsize 1 \
        --init_coeff -0.25 \
        --iteration 3 \
        --add_prefix \
        --return_hiddens \
        --max_norm 0.8

    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-final-gogogo \
        --start_from_idx 500 \
        --max_num_data 100 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --do_sample \
        --n_branches 3 \
        --batchsize 1 \
        --init_coeff -0.25 \
        --iteration 3 \
        --binary \
        --add_prefix \
        --return_hiddens \
        --max_norm 0.8

    CUDA_VISIBLE_DEVICES=6 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-suffix \
        --start_from_idx 0 \
        --max_num_data 500 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --do_sample \
        --batchsize 1 \
        --binary \
        --init_coeff -0.25 \
        --iteration 3 \
        --return_hiddens \
        --add_prefix \
        --max_norm 0.8

done