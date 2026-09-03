"""Default hyperparameter settings for the inter-case Camargo LSTM pipeline.

Identical search space to camargo/params.py -- only difference is
model_type, which points at the two new [shared_cat_ic]/[concatenated_ic]
sections added to models_spec.ini (additional_columns = the 7 inter-case
load-state features instead of [daytime, weekday]). Same lstm_act /
learning_rate / dropout / n_size / l_size choices, same epochs/max_eval
defaults as the baseline camargo HPO (see pipelines/camargo/camargo_pipeline.ipynb),
so the two HPOs get an equal-sized search.
"""


def default_params(run_name: str, *, max_eval: int = 10, epochs: int = 200) -> dict:
    return {
        # identity
        "file_name": f"{run_name}.csv",

        # preprocessing
        "rp_sim": 0.85,
        "one_timestamp": True,

        # training loop
        "batch_size": 32,
        "epochs": epochs,
        "max_eval": max_eval,
        "imp": 1,

        # HPO search space -- same as camargo/params.py, model_type swapped
        # for the inter-case-feature-carrying variants.
        "model_type":     ["shared_cat_ic", "concatenated_ic"],
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
