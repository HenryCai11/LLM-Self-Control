cd ..

lr_list=(3e-3 4e-3 5e-3 6e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-2k \
        --eval_set_name happy2sad-eval \
        --attribute happy2sad \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 16 \
        --peft_type "llama-adapter" \
        --searching \
        --max_num_data 500\
        --pick_by_eval 

    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-2k \
        --eval_set_name happy2sad-eval \
        --attribute happy2sad \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 16 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --do_test

done