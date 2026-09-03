"""Stub for analyzers.sim_evaluator — evaluation metrics not used in this setup."""
import pandas as pd


class Evaluator:
    def __init__(self, one_timestamp):
        pass

    def measure(self, method, data, attr):
        return pd.DataFrame()


class SimilarityEvaluator:
    def __init__(self, log, sim_log, parms):
        self.similarity = {}

    def measure_distance(self, metric):
        self.similarity = {"metric": metric, "value": float("nan")}
