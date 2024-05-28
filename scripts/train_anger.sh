cd ..

CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name angry2peaceful-final \
    --eval_set_name angry2peaceful-eval-final \
    --attribute angry2peaceful \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --max_num_data 2000 \
    --pick_by_eval 

CUDA_VISIBLE_DEVICES=5 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name angry2peaceful-final \
    --eval_set_name angry2peaceful-eval-final \
    --attribute angry2peaceful \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test