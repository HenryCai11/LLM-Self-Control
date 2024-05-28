cd ..

CUDA_VISIBLE_DEVICES=6 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name privacy-final-final \
    --eval_set_name privacy-eval-final-final \
    --attribute privacy \
    --batchsize 8 \
    --lr 3e-3 \
    --accumulation_steps 16 \
    --peft_type "prefix+adapter" \
    --max_num_data 800 \
    --pick_by_eval \
    --norm_threshold 0.2 \

CUDA_VISIBLE_DEVICES=6 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name privacy-final-final \
    --eval_set_name privacy-eval-final-final \
    --attribute privacy \
    --batchsize 8 \
    --lr 3e-3 \
    --accumulation_steps 16 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test