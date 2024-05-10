cd ..

lr_list=(3e-4 2e-4 1e-4)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=2 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-1k-search-3iter \
        --eval_set_name happy2sad-eval-search-3iter \
        --attribute happy2sad \
        --batchsize 1 \
        --lr $lr \
        --accumulation_steps 1 \
        --peft_type "llama-adapter" \
        --max_num_data 1000 \
        --pick_by_eval \
        --searching

    CUDA_VISIBLE_DEVICES=2 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-1k-search-3iter \
        --eval_set_name happy2sad-eval-search-3iter \
        --attribute happy2sad \
        --batchsize 1 \
        --lr $lr \
        --accumulation_steps 1 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --do_test

done