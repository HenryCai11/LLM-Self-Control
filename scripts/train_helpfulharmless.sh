cd ..

lr_list=(1e-3)

for lr in "${lr_list[@]}"
do
    CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
        --training_set_name helpfulharmless-final-gogogo \
        --eval_set_name helpfulharmless-eval-suffix \
        --attribute helpfulharmless \
        --batchsize 16 \
        --lr $lr \
        --accumulation_steps 4 \
        --peft_type "prefix+adapter" \
        --max_num_data 2000\
        --pick_by_eval

    CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
        --training_set_name helpfulharmless-final-gogogo \
        --eval_set_name helpfulharmless-eval-suffix \
        --attribute helpfulharmless \
        --batchsize 16 \
        --lr $lr \
        --accumulation_steps 4 \
        --peft_type "prefix+adapter" \
        --pick_by_eval \
        --do_test

done