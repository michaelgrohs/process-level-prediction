"""Default hyperparameter settings for the ProcessTransformer (Bukhsh et al.) pipeline."""


def default_params(*, epochs: int = 50, batch_size: int = 12,
                   learning_rate: float = 0.001, num_heads: int = 4) -> dict:
    return {
        "epochs":           epochs,
        "batch_size":       batch_size,
        "learning_rate":    learning_rate,
        "num_heads":        num_heads,
        "rp_sim":           0.85,
        "time_format":      "%Y-%m-%d %H:%M:%S",
        "max_suffix_length": 50,
    }
