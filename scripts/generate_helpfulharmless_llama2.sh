cd ..
attribute_list=("helpfulharmless")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=6 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-final-inst \
        --start_from_idx 0 \
        --max_num_data 800 \
        --max_new_tokens 256 \
        --epoch 1 \
        --search \
        --max_num_data 256 \
        --batchsize 1 \
        --init_coeff -0.5 \
        --iteration 2 \
        --return_hiddens \
        --add_inst \
        --max_norm 0.5

    # CUDA_VISIBLE_DEVICES=6 python -m self_control.suffix_gradient.generate_delta_ds \
    #     --attribute ${attribute_list[$i]} \
    #     --output_name ${attribute_list[$i]}-eval-final-inst \
    #     --start_from_idx 0 \
    #     --max_num_data 100 \
    #     --max_new_tokens 256 \
    #     --epoch 1 \
    #     --search \
    #     --batchsize 1 \
    #     --init_coeff -0.5 \
    #     --iteration 3 \
    #     --return_hiddens \
    #     --add_inst \
    #     --max_norm 0.5

done