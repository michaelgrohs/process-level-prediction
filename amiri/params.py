def default_params(**overrides) -> dict:
    params = {
        # GPS architecture
        "gt_layers": 5,
        "gt_n_heads": 8,
        "gt_dim_hidden": 64,
        "gt_dropout": 0.2,
        "gt_attn_dropout": 0.5,
        # Positional encodings
        "posenc_lapPE_dim_pe": 8,
        "posenc_rwse_dim_pe": 8,
        "posenc_rwse_k": 36,
        # Optimizer
        "base_lr": 0.0005,
        "weight_decay": 1e-2,
        # Training
        "batch_size": 128,
        "max_epoch": 100,
        "seed": 42,
    }
    params.update(overrides)
    return params
