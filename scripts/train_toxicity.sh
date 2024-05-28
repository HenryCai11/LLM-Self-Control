cd ..

lr_list=(3e-3)

for lr in "${lr_list[@]}"
do
    # CUDA_VISIBLE_DEVICES=3 python -m self_control.prefix_control.prefix_trainer \
    #     --training_set_name toxic2nontoxic-final-new \
    #     --eval_set_name toxic2nontoxic-eval-final-new \
    #     --attribute toxic2nontoxic \
    #     --batchsize 16 \
    #     --lr $lr \
    #     --accumulation_steps 8 \
    #     --peft_type "prefix+adapter" \
    #     --max_num_data 2000\
    #     --pick_by_eval

    CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
        --training_set_name toxic2nontoxic-final-new \
        --eval_set_name toxic2nontoxic-eval-final-new \
        --attribute toxic2nontoxic \
        --batchsize 16 \
        --lr $lr \
        --accumulation_steps 8 \
        --peft_type "prefix+adapter" \
        --pick_by_eval \
        --do_test

done