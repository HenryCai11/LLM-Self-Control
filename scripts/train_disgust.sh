cd ..

CUDA_VISIBLE_DEVICES=7 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name disgusted2satisfied-final \
    --eval_set_name disgusted2satisfied-eval-final \
    --attribute disgusted2satisfied \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --max_num_data 800 \
    --pick_by_eval 

CUDA_VISIBLE_DEVICES=7 python -m self_control.prefix_control.prefix_trainer \
    --training_set_name disgusted2satisfied-final \
    --eval_set_name disgusted2satisfied-eval-final \
    --attribute disgusted2satisfied \
    --batchsize 32 \
    --lr 3e-3 \
    --accumulation_steps 4 \
    --peft_type "prefix+adapter" \
    --pick_by_eval \
    --do_test