cd ..

lr_list=(3e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --training_set_name reasoning-smallernorm-final-gogogo \
        --eval_set_name reasoning-eval-final-gogogo \
        --attribute reasoning \
        --batchsize 8 \
        --lr $lr \
        --accumulation_steps 16 \
        --peft_type "prefix+adapter" \
        --max_num_data 2000\
        --norm_threshold 0.5 \
        --pick_by_eval

    CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
        --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
        --training_set_name reasoning-smallernorm-final-gogogo \
        --eval_set_name reasoning-eval-final-gogogo \
        --attribute reasoning \
        --batchsize 8 \
        --lr $lr \
        --accumulation_steps 16 \
        --peft_type "prefix+adapter" \
        --pick_by_eval \
        --norm_threshold 0.5 \
        --do_test

done