"""
Copyright 2024 Universitat Politècnica de Catalunya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import tensorflow as tf
import pickle
import os


def get_fermi_z_scores(
    ds,
    params,
    flatten=False,
    summarize=-1,
    only_positive=False,
    store_res_path=None,
    check_existing=False,
):
    """
    Get the mean and the std for different parameters of a dataset. This function is
    meant to be used by the Fermi model, which performs z-score normalization just
    before the input layer (unlike wavelet methods, which perform it in the
    preprocessing step).

    Parameters
    ----------
    ds
        tensorflow.data.Dataset
    params
        list of strings indicating the parameters to extract the features from
    include_y, optional
        Indicates if to also extract the features of the output variable. Inputs
        indicate the string key used on the return dict. If None, it is not included.
    flatten, optional
        If true, mean and std are computed globally for all dimensions in each feature.
        Otherwise, the values are computed for each dimension separately.
    summarize, optional
        If > 0, only uses the first n samples to compute the mean and std.
    store_res_path, optional
        If not None, the results are stored in the path indicated by the string.
        The dictionary is stored using the pickle library.
    check_existing, optional
        If True, check if the file indicated by store_res_path exists and return its
        content. Otherwise, compute the mean and std as usual.

    Returns
    -------
    dict
        Dictionary containing the min and the max-min for each parameter.
    """
    # If check_existing is True, check if the file exists and return the dict (if so)
    if store_res_path is not None and check_existing:
        if os.path.exists(store_res_path):
            with open(store_res_path, "rb") as ff:
                return pickle.load(ff)

    def _only_positive(x):
        if not only_positive:
            return x
        # else
        return x[x > 0]

    # Use first sample to get the shape of the tensors
    iter_ds = iter(ds)
    next_sample = next(iter_ds)
    sample, label = next_sample[0], next_sample[1]
    params_dims = {param: sample[param].numpy().shape[-1] for param in params}
    params_lists = {
        param: (
            _only_positive(sample[param].numpy()).flatten()
            if flatten
            else _only_positive(
                tf.reshape(sample[param], (-1, params_dims[param])).numpy()
            )
        )
        for param in params
    }

    if summarize > 0:
        max_samples = summarize - 1

    # Include the rest of the samples
    for ii, (sample, label) in enumerate(map(lambda x: (x[0], x[1]), iter_ds)):
        if summarize > 0 and ii > max_samples:
            break
        for param in params:
            new_val = _only_positive(
                tf.reshape(sample[param], (-1, params_dims[param])).numpy()
            )
            if flatten:
                new_val = new_val.flatten()
            params_lists[param] = np.concatenate((params_lists[param], new_val), axis=0)

    scores = dict()
    axis = None if flatten else 0
    for param, param_list in params_lists.items():
        scores[param] = [np.mean(param_list, axis=axis), np.std(param_list, axis=axis)]
        # Check if std is 0
        if scores[param][1].size == 1 and scores[param][1] == 0:
            print(f"Z-score normalization Warning: {param} has a std of 0.")
            scores[param][1] = 1
        elif scores[param][1].size > 1 and np.any(scores[param][1] == 0):
            print(
                f"Z-score normalization Warning: Several values of {param} has a std of 0."
            )
            scores[param][1][scores[param][1] == 0] = 1

    if store_res_path is not None:
        store_res_dir, _ = os.path.split(store_res_path)
        os.makedirs(store_res_dir, exist_ok=True)
        with open(store_res_path, "wb") as ff:
            pickle.dump(scores, ff)

    return scores


def log_transform(x, y):
    """Apply log transformation to output variable.

    Parameters
    ----------
    x: dict
        Predictor variables
    y: tf.Tensor
        Output variable

    Returns
    -------
    dict
        Predictor variables
    tf.Tensor
        Transformed output variable
    """
    return x, tf.math.log(y)


def denorm_mape_metric(y_true, y_pred):
    """
    Obtains the MAPE metric when the output variable is log-transformed.

    Parameters
    ----------
    y_true: tf.Tensor
        Ground truth
    y_pred: tf.Tensor
        Output variable

    Returns
    -------
    float
        MAPE metric
    """

    y_true = tf.math.exp(y_true)
    y_pred = tf.math.exp(y_pred)
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true)) * 100
