cd ..

# CUDA_VISIBLE_DEVICES=3 python -m self_control.prefix_control.adapter_no_trainer \
#     --training_set_name privacy-final \
#     --eval_set_name privacy-eval-final \
#     --attribute privacy \
#     --batchsize 8 \
#     --lr 3e-3 \
#     --accumulation_steps 16 \
#     --peft_type "prefix+adapter" \
#     --max_num_data 800 \
#     --pick_by_eval 

CUDA_VISIBLE_DEVICES=2 python -m self_control.prefix_control.adapter_no_trainer \
    --training_set_name privacy-final \
    --eval_set_name privacy-eval-final \
    --attribute privacy \
    --batchsize 8 \
    --lr 3e-3 \
    --accumulation_steps 16 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test