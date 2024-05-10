cd ..

lr_list=(3e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name toxic2nontoxic-100 \
        --eval_set_name toxic2nontoxic-eval \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --max_num_data 100 \
        --name_prefix "checkpoint-study" \
        --accumulation_steps 1 \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --searching

    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.adapter_no_trainer \
        --training_set_name toxic2nontoxic-100 \
        --eval_set_name toxic2nontoxic-eval \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 1 \
        --name_prefix "checkpoint-study" \
        --peft_type "llama-adapter" \
        --pick_by_eval \
        --do_test
        # --test_original \

done