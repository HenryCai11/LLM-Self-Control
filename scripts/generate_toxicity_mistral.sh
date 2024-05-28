cd ..
attribute_list=("toxic2nontoxic")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]}\
        --output_name ${attribute_list[$i]}-final-mistral \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 5 \
        --search \
        --do_sample \
        --batchsize 1 \
        --init_coeff -0.1 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --max_norm 1

    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final-mistral \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 1 \
        --search \
        --batchsize 1 \
        --init_coeff -0.1 \
        --iteration 2 \
        --return_hiddens \
        --max_norm 1 \
        --add_prefix
done