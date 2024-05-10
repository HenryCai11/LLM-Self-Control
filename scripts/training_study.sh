# This script is for studying what leads to 
cd ..

# no search
CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
    --training_set_name happy2sad-2k \
    --eval_set_name happy2sad-eval \
    --attribute happy2sad \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 8 \
    --peft_type "llama-adapter" \
    --max_num_data 2000 \
    --name_prefix "study-" \
    --pick_by_eval 

CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
    --training_set_name happy2sad-2k \
    --eval_set_name happy2sad-eval \
    --attribute happy2sad \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 8 \
    --peft_type "llama-adapter" \
    --pick_by_eval \
    --name_prefix "study-eval-" \
    --do_test

# search
# different data size
# size_list=(2000 1500 1000 500 100)
size_list=(1000 500 100)

for size in "${size_list[@]}"
do
    CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-2k-search \
        --eval_set_name happy2sad-eval-search \
        --attribute happy2sad \
        --batchsize 32 \
        --lr 3e-3 \
        --accumulation_steps 8 \
        --peft_type "llama-adapter" \
        --max_num_data $size \
        --name_prefix "study-" \
        --pick_by_eval 

    CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name happy2sad-2k-search \
        --eval_set_name happy2sad-eval-search \
        --attribute happy2sad \
        --batchsize 32 \
        --lr 3e-3 \
        --accumulation_steps 8 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --name_prefix "study-eval-" \
        --do_test

done

# search & filter by suffix score
CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
    --training_set_name happy2sad-2k-search-filtered \
    --eval_set_name happy2sad-eval-search \
    --attribute happy2sad \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 8 \
    --peft_type "llama-adapter" \
    --max_num_data 2000 \
    --name_prefix "study-"
    # --pick_by_eval 

# CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
#     --training_set_name happy2sad-2k-search-filtered \
#     --eval_set_name happy2sad-eval-search \
#     --attribute happy2sad \
#     --batchsize 32 \
#     --lr 3e-3 \
#     --accumulation_steps 8 \
#     --peft_type "llama-adapter" \
#     --pick_by_eval \
#     --name_prefix "study-eval-" \
#     --do_test