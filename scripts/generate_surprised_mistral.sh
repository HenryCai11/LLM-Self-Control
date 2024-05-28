cd ..
emo_data=("surprise")
attribute_list=("calm2surprised")

for ((i = 0; i < 1; i++));
do
    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-final-mistral \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 8 \
        --search \
        --do_sample \
        --batchsize 1 \
        --init_coeff -0.1 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --binary \
        --add_inst \
        --max_norm 1 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final-mistral \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 2 \
        --search \
        --do_sample \
        --batchsize 1 \
        --init_coeff -0.1 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --binary \
        --add_inst \
        --max_norm 1 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json
done