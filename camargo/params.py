"""Default hyperparameter settings for Camargo LSTM pipeline."""


def default_params(run_name: str, *, max_eval: int = 5, epochs: int = 200) -> dict:
    """
    Returns a params dict with sensible defaults. Override any key before
    passing to CamargoTrainer.

    run_name is used as the embedding file stem and the output folder name.
    """
    return {
        # identity
        "file_name": f"{run_name}.csv",   # used to name .emb files

        # preprocessing
        "rp_sim": 0.85,                   # resource-pool similarity threshold
        "one_timestamp": True,            # True if log only has end_timestamp

        # training loop
        "batch_size": 32,
        "epochs": epochs,
        "max_eval": max_eval,             # Bayesian HPO trials
        "imp": 1,

        # HPO search space  (lists → hyperopt hp.choice; single-element = fixed)
        "model_type":     ["shared_cat", "concatenated"],
        "lstm_act":       ["selu", "tanh"],
        "learning_rate":  [0.0005, 0.001, 0.002],
        "dropout":        [0.1, 0.2, 0.3],
        # fixed params (single-element, not really searched)
        "norm_method":    ["max"],
        "n_size":         [10],
        "l_size":         [100],
        "dense_act":      ["linear"],
        "optim":          ["Nadam"],

        # needed by ModelPredictor internals
        "read_options": {
            "timeformat": "%Y-%m-%dT%H:%M:%S.%f",
            "one_timestamp": True,
        },
    }
