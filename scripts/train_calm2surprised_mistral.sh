cd ..

# CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.prefix_trainer \
#     --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
#     --training_set_name calm2surprised-final-mistral \
#     --eval_set_name calm2surprised-eval-final-mistral \
#     --attribute calm2surprised \
#     --batchsize 16 \
#     --lr 3e-3 \
#     --accumulation_steps 8 \
#     --peft_type "prefix+adapter" \
#     --max_num_data 800 \
#     --pick_by_eval 

CUDA_VISIBLE_DEVICES=4 python -m self_control.prefix_control.prefix_trainer \
    --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
    --training_set_name calm2surprised-final-mistral \
    --eval_set_name calm2surprised-eval-final-mistral \
    --attribute calm2surprised \
    --batchsize 16 \
    --lr 3e-3 \
    --accumulation_steps 8 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test