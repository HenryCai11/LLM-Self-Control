cd ..

# CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
#     --training_set_name surprised2calm-final \
#     --eval_set_name surprised2calm-eval-final \
#     --attribute surprised2calm \
#     --batchsize 16 \
#     --lr 3e-3 \
#     --accumulation_steps 8 \
#     --peft_type "prefix+adapter" \
#     --max_num_data 800 \
#     --pick_by_eval 

CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name surprised2calm-final \
    --eval_set_name surprised2calm-eval-final \
    --attribute surprised2calm \
    --batchsize 16 \
    --lr 3e-3 \
    --accumulation_steps 8 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test