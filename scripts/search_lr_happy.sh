cd ..

lr_list=(3e-3 4e-3 5e-3 6e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-1k-search-noshuffle \
        --eval_set_name happy2sad-eval-search-noshuffle \
        --attribute happy2sad \
        --batchsize 16 \
        --lr $lr \
        --accumulation_steps 1 \
        --peft_type "llama-adapter" \
        --searching \
        --max_num_data 1500 \
        --pick_by_eval 

    CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-1k-search-noshuffle \
        --eval_set_name happy2sad-eval-search-noshuffle \
        --attribute happy2sad \
        --batchsize 16 \
        --lr $lr \
        --accumulation_steps 1 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --do_test

done