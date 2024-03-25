deepspeed_config = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": [0.9, 0.999],
            "eps": "auto",
            "weight_decay": "auto",
            "torch_adam": True,
            "adam_w_mode": True
        }
    },

    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 1e-6,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            
        }
    },

    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True
    },
    
    "sparse_attention": {
        "mode": "fixed",
        "block": 16,
        "different_layout_per_head": True,
        "num_local_blocks": 4,
        "num_global_blocks": 1,
        "attention": "bidirectional",
        "horizontal_global_attention": False,
        "num_different_global_patterns": 4,
        "num_random_blocks": 0,
        "local_window_blocks": [4],
        "global_block_indices": [0],
        "global_block_end_indices": None,
        "num_sliding_window_blocks": 3
      },

    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
    "steps_per_print": 2000,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": False,
}