cd ..
attribute_list=("truthful")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-final \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 10 \
        --search \
        --do_sample \
        --batchsize 2 \
        --init_coeff -2 \
        --iteration 2 \
        --return_hiddens \
        --add_inst \
        --add_prefix \
        --max_norm 1

    CUDA_VISIBLE_DEVICES=0 python -m self_control.suffix_gradient.generate_delta_ds \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 1 \
        --search \
        --do_sample \
        --batchsize 2 \
        --init_coeff -2 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 1 \
        --add_inst \
        --add_prefix

done