cd ..
emo_data=("happiness" "anger" "fear" "surprise" "disgust")
attribute_list=("happy2sad" "angry2peaceful" "afraid2fearless" "surprised2calm" "disgusted2satisfied")

for ((i = 0; i < 5; i++));
do
    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-final-mistral \
        --start_from_idx 0 \
        --max_num_data 100 \
        --epoch 20 \
        --search \
        --do_sample \
        --batchsize 2 \
        --init_coeff -1 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --binary \
        --add_inst \
        --max_norm 0.5 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json

    CUDA_VISIBLE_DEVICES=5 python -m self_control.suffix_gradient.generate_delta_ds \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --attribute ${attribute_list[$i]} \
        --output_name ${attribute_list[$i]}-eval-final-mistral \
        --start_from_idx 100 \
        --max_num_data 100 \
        --epoch 3 \
        --search \
        --do_sample \
        --batchsize 2 \
        --init_coeff -1 \
        --iteration 2 \
        --return_hiddens \
        --add_prefix \
        --binary \
        --add_inst \
        --max_norm 0.5 \
        --data_path benchmarks/emotions/${emo_data[$i]}.json
done