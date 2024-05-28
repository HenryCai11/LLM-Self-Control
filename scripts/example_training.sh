cd ..

lr_list=(3e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.prefix_trainer \
        --training_set_name toxic2nontoxic-final-2iter-02norm \
        --eval_set_name toxic2nontoxic-eval-final-2iter-02norm \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 4 \
        --peft_type "prefix+adapter" \
        --max_num_data 1500 \
        --pick_by_eval 

    CUDA_VISIBLE_DEVICES=1 python -m self_control.prefix_control.prefix_trainer \
        --training_set_name toxic2nontoxic-final-2iter-02norm \
        --eval_set_name toxic2nontoxic-eval-final-2iter-02norm \
        --attribute toxic2nontoxic \
        --batchsize 32 \
        --lr $lr \
        --accumulation_steps 4 \
        --peft_type "prefix+adapter" \
        --pick_by_eval \
        --do_test

done