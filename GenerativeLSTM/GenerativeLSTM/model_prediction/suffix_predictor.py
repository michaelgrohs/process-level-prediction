# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:35:37 2020

@author: Manuel Camargo
"""
import os
import pickle

import numpy as np

from tensorflow.keras.models import load_model

# Keras version-mismatch fix: models saved with a Keras version that serialises
# `quantization_config` in every layer config fail to load when that kwarg is
# unknown in the local Keras. Patch the base Layer class to silently drop it.
try:
    from keras.src.layers.layer import Layer as _KLayer
    _orig_layer_init = _KLayer.__init__
    def _compat_layer_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        _orig_layer_init(self, *args, **kwargs)
    _KLayer.__init__ = _compat_layer_init
except Exception:
    pass

import utils.support as sup


class SuffixPredictor():

    def __init__(self):
        """constructor"""
        self.model = None
        self.spl = dict()
        self.imp = 'arg_max'
        self.max_trace_size = 0

    def predict(self, params, model, spl, imp, vectorizer):
        self.model = load_model(model, compile=False)
        self.spl = spl
        self.max_trace_size = params['max_trace_size']
        self.imp = imp
        predictor = self._get_predictor(params['model_type'])
        sup.print_performed_task('Predicting suffixes')
        return predictor(params, vectorizer)

    def _get_predictor(self, model_type):
        # OJO: This is an extension point just incase
        # a different predictor being neccesary
        return self._predict_suffix_shared_cat

    def _prefixes_signature(self):
        """Cheap fingerprint of self.spl['prefixes']['activities'] -- total
        count plus the per-prefix length of every entry, hashed. Two runs
        over genuinely the same case list (same order, same content) always
        produce the same signature; almost any real difference (different
        dataset, different split, reordered/truncated cases) changes it. Used
        to refuse resuming a checkpoint that doesn't match the current run --
        see the checkpoint-resume comment in _predict_suffix_shared_cat."""
        import hashlib
        acts = self.spl['prefixes']['activities']
        lengths = ','.join(str(len(p)) for p in acts)
        return hashlib.sha1(f'{len(acts)}|{lengths}'.encode()).hexdigest()

    def _predict_suffix_shared_cat(self, parms, vectorizer):
        """Generate business process suffixes using a keras trained model.
        Args:
            model (keras model): keras trained model.
            prefixes (list): list of prefixes.
            ac_index (dict): index of activities.
            rl_index (dict): index of roles.
            imp (str): method of next event selection.
        """
        # Generation of predictions — resume from checkpoint if one exists.
        # `next_i` is a purely positional index into self.spl['prefixes'], so a
        # checkpoint is only safe to resume from if it was saved against the
        # EXACT SAME prefix list this run is about to iterate — otherwise `next_i`
        # silently skips over the wrong cases (confirmed happening: a stale
        # checkpoint for a differently-sized run caused ~2259 fresh test cases to
        # be silently dropped from a regenerated full-regime prediction, with no
        # error — see project CLAUDE.md, 2026-08-21). `_prefixes_signature` is a
        # cheap (count, per-prefix-length tuple) fingerprint of the exact prefix
        # list — mismatched signature means "not the same run", so the checkpoint
        # is discarded and prediction starts fresh rather than silently resuming
        # from the wrong position.
        checkpoint_path = parms.get('checkpoint_path')
        checkpoint_interval = parms.get('checkpoint_interval', 500)
        current_sig = self._prefixes_signature()
        results = list()
        start_i = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as _f:
                _ckpt = pickle.load(_f)
            ckpt_sig = _ckpt.get('prefixes_signature')
            if ckpt_sig is not None and ckpt_sig != current_sig:
                print(f'[checkpoint] IGNORING stale/incompatible checkpoint at {checkpoint_path} '
                     f'(saved against a different prefix list than this run -- '
                     f'checkpoint sig {ckpt_sig[:2]}... != current sig {current_sig[:2]}...). '
                     f'Starting fresh instead of silently dropping cases.')
            else:
                results = _ckpt['results']
                start_i = _ckpt['next_i']
                print(f'[checkpoint] Resuming from trace {start_i} ({len(results)} already done)')

        for i, _ in enumerate(self.spl['prefixes']['activities']):
            if i < start_i:
                continue
            # Activities and roles input shape(1,5)
            x_ac_ngram = (np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['activities'][i]),
                    axis=0)[-parms['dim']['time_dim']:]
                .reshape((1, parms['dim']['time_dim'])))

            x_rl_ngram = (np.append(
                    np.zeros(parms['dim']['time_dim']),
                    np.array(self.spl['prefixes']['roles'][i]),
                    axis=0)[-parms['dim']['time_dim']:]
                .reshape((1, parms['dim']['time_dim'])))
           
            times_attr_num = (self.spl['prefixes']['times'][i].shape[1])
            x_t_ngram = np.array(
                [np.append(np.zeros(
                    (parms['dim']['time_dim'], times_attr_num)),
                    self.spl['prefixes']['times'][i], axis=0)
                    [-parms['dim']['time_dim']:]
                    .reshape((parms['dim']['time_dim'], times_attr_num))]
                )
            if vectorizer in ['basic']:
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
            elif vectorizer in ['inter']:
                inter_attr_num = self.spl['prefixes']['inter_attr'][i].shape[1]
                x_inter_ngram = np.array([np.append(
                        np.zeros((parms['dim']['time_dim'], inter_attr_num)),
                        self.spl['prefixes']['inter_attr'][i],
                        axis=0)[-parms['dim']['time_dim']:].reshape((parms['dim']['time_dim'], inter_attr_num))])
                inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]

            pref_size = len(self.spl['prefixes']['activities'][i])
            acum_dur, acum_wait = list(), list()
            ac_suf, rl_suf = list(), list()
            for _  in range(1, self.max_trace_size):
                preds = self.model.predict(inputs)
                if self.imp == 'random_choice':
                    # Use this to get a random choice following as PDF the predictions
                    pos = np.random.choice(
                        np.arange(0,len(preds[0][0])), p=preds[0][0])
                    pos1 = np.random.choice(
                        np.arange(0, len(preds[1][0])), p=preds[1][0])
                elif self.imp == 'arg_max':
                    # Use this to get the max prediction
                    pos = np.argmax(preds[0][0])
                    pos1 = np.argmax(preds[1][0])
                # Activities accuracy evaluation
                x_ac_ngram = np.append(x_ac_ngram, [[pos]], axis=1)
                x_ac_ngram = np.delete(x_ac_ngram, 0, 1)
                x_rl_ngram = np.append(x_rl_ngram, [[pos1]], axis=1)
                x_rl_ngram = np.delete(x_rl_ngram, 0, 1)
                x_t_ngram = np.append(x_t_ngram, [preds[2]], axis=1)
                x_t_ngram = np.delete(x_t_ngram, 0, 1)
                if vectorizer in ['basic']:
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram]
                elif vectorizer in ['inter']:
                    x_inter_ngram = np.append(x_inter_ngram, [preds[3]], axis=1)
                    x_inter_ngram = np.delete(x_inter_ngram, 0, 1)
                    inputs = [x_ac_ngram, x_rl_ngram, x_t_ngram, x_inter_ngram]
                # Stop if the next prediction is the end of the trace
                # otherwise until the defined max_size
                ac_suf.append(int(pos))
                rl_suf.append(int(pos1))
                acum_dur.append(float(preds[2][0][0]))
                if not parms['one_timestamp']:
                    acum_wait.append(float(preds[2][0][1]))
                if parms['index_ac'][pos] == 'end':
                    break
            # save results
            predictions = [ac_suf, rl_suf, acum_dur]
            if not parms['one_timestamp']:
                predictions.extend([acum_wait])
            results.append(
                self.create_result_record(i, self.spl, predictions, parms, pref_size))

            if checkpoint_path and (len(results) % checkpoint_interval == 0):
                with open(checkpoint_path, 'wb') as _f:
                    pickle.dump({'results': results, 'next_i': i + 1,
                                'prefixes_signature': current_sig}, _f)

        if checkpoint_path and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        sup.print_done_task()
        return results

    def create_result_record(self, index, spl, preds, parms, pref_size):
        record = dict()
        record['caseid'] = spl['prefixes']['caseids'][index]
        record['pref_size'] = pref_size
        record['ac_prefix'] = spl['prefixes']['activities'][index]
        record['ac_expect'] = spl['next_evt']['activities'][index]
        record['ac_pred'] = preds[0]
        record['rl_prefix'] = spl['prefixes']['roles'][index]
        record['rl_expect'] = spl['next_evt']['roles'][index]
        record['rl_pred'] = preds[1]
        if parms['one_timestamp']:
            record['tm_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']) 
                for x in spl['prefixes']['times'][index]]
            record['tm_expect'] = [self.rescale(
                x[0], parms, parms['scale_args']) 
                for x in spl['next_evt']['times'][index]]
            record['tm_pred'] = [self.rescale(
                x, parms, parms['scale_args']) 
                for x in preds[2]]
        else:
            # Duration
            record['dur_prefix'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur']) 
                for x in spl['prefixes']['times'][index]]
            record['dur_expect'] = [self.rescale(
                x[0], parms, parms['scale_args']['dur']) 
                for x in spl['next_evt']['times'][index]]
            record['dur_pred'] = [self.rescale(
                x, parms, parms['scale_args']['dur']) 
                for x in preds[2]]
            # Waiting
            record['wait_prefix'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait']) 
                for x in spl['prefixes']['times'][index]]
            record['wait_expect'] = [self.rescale(
                x[1], parms, parms['scale_args']['wait']) 
                for x in spl['next_evt']['times'][index]]
            record['wait_pred'] = [self.rescale(
                x, parms, parms['scale_args']['wait']) 
                for x in preds[3]]
        return record

    @staticmethod
    def rescale(value, parms, scale_args):
        if parms['norm_method'] == 'lognorm':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
            value = np.expm1(value)
        elif parms['norm_method'] == 'normal':
            max_value = scale_args['max_value']
            min_value = scale_args['min_value']
            value = (value * (max_value - min_value)) + min_value
        elif parms['norm_method'] == 'standard':
            mean = scale_args['mean']
            std = scale_args['std']
            value = (value * std) + mean
        elif parms['norm_method'] == 'max':
            max_value = scale_args['max_value']
            value = np.rint(value * max_value)
        elif parms['norm_method'] is None:
            value = value
        else:
            raise ValueError(parms['norm_method'])
        return value
