cd ..
attribute_list=("reasoning")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-llama-final-gogogo \
        --start_from_idx 0 \
        --max_num_data 400 \
        --epoch 1 \
        --max_new_tokens 256 \
        --search \
        --batchsize 1 \
        --init_coeff -0.1 \
        --n_branches 6 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --max_norm 0.5

    CUDA_VISIBLE_DEVICES=3 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-llama-final-gogogo \
        --start_from_idx 0 \
        --max_num_data 100 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --batchsize 1 \
        --init_coeff -0.1 \
        --n_branches 6 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 0.5 \
        --add_prefix

done