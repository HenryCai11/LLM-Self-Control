cd ..
attribute_list=("helplessharmful")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=6 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-gogogo \
        --start_from_idx 0 \
        --max_num_data 200 \
        --epoch 1 \
        --search \
        --do_sample \
        --n_branches 3 \
        --max_new_tokens 256 \
        --batchsize 1 \
        --binary \
        --init_coeff -0.25 \
        --iteration 3 \
        --return_hiddens \
        --max_norm 0.5

    CUDA_VISIBLE_DEVICES=6 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-gogogo \
        --start_from_idx 400 \
        --max_num_data 200 \
        --epoch 1 \
        --search \
        --do_sample \
        --n_branches 3 \
        --max_new_tokens 256 \
        --binary \
        --batchsize 1 \
        --init_coeff -0.25 \
        --iteration 3 \
        --return_hiddens \
        --max_norm 0.5

    CUDA_VISIBLE_DEVICES=7 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final-inst \
        --start_from_idx 0 \
        --max_num_data 100 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --batchsize 1 \
        --init_coeff -0.5 \
        --iteration 3 \
        --return_hiddens \
        --add_inst \
        --max_norm 0.5

done