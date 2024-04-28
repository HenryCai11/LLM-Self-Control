cd ..

lr_list=(3e-3 1e-3 9e-4 6e-3)
# for lr in "${lr_list[@]}"
# do
#     CUDA_VISIBLE_DEVICES=0 python -m self_control.prefix_control.adapter_no_trainer \
#         --training_set_name toxic2nontoxic-100 \
#         --eval_set_name toxic2nontoxic-eval \
#         --attribute toxic2nontoxic \
#         --batchsize 32 \
#         --lr $lr \
#         --accumulation_steps 16 \
#         --peft_type "lora" \
#         --pick_by_eval \
#         --searching \

#     CUDA_VISIBLE_DEVICES=0 python -m self_control.prefix_control.adapter_no_trainer \
#         --training_set_name toxic2nontoxic-100 \
#         --eval_set_name toxic2nontoxic-eval \
#         --attribute toxic2nontoxic \
#         --batchsize 32 \
#         --lr $lr \
#         --accumulation_steps 16 \
#         --peft_type "lora" \
#         --pick_by_eval \
#         --do_test

# done

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name toxic2nontoxic-100-filtered \
        --eval_set_name toxic2nontoxic-eval \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 8 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --searching \

    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name toxic2nontoxic-100-filtered \
        --eval_set_name toxic2nontoxic-eval \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 8 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --do_test

done