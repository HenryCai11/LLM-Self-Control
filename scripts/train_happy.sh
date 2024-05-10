cd ..

# CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.adapter_no_trainer \
#     --training_set_name happy2sad-2iter-02norm \
#     --eval_set_name happy2sad-eval-2iter-02norm \
#     --attribute happy2sad \
#     --batchsize 32 \
#     --lr 3e-3 \
#     --accumulation_steps 4 \
#     --peft_type "prefix+adapter" \
#     --max_num_data 2000 \
#     --pick_by_eval 

CUDA_VISIBLE_DEVICES=7 python -m self_control.prefix_control.adapter_no_trainer \
    --training_set_name happy2sad-2iter-02norm \
    --eval_set_name happy2sad-eval-2iter-02norm \
    --attribute happy2sad \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test