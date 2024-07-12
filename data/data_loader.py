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

import tensorflow as tf
import os
from pickle import load


def load_partition(path):
    """Loads a partition of the dataset.

    Parameters
    ----------
    path : str
        Path to the partition directory.

    Returns
    -------
    tf.data.Dataset
        Loaded partition
    """
    num_partitions = len(os.listdir(path))
    ds = tf.data.Dataset.load(os.path.join(path, "0000"), compression="GZIP")
    for ii in range(1, num_partitions):
        ds = ds.concatenate(
            tf.data.Dataset.load(os.path.join(path, f"{ii:04d}"), compression="GZIP")
        )
    return ds


def load_z_scores(path):
    """Returns the mean and std of the training dataset to compute the z-scores.

    Returns
    -------
    (float, float, float, float)
        Returns the mean and std of the IPG, the mean and std of the packet size,
        in that order.
    """
    with open(path, "rb") as ff:
        return load(ff)


# def fix_time_dist(x, y):
#     x["flow_time_dist"] = tf.where(
#         tf.math.equal(x["flow_time_dist"], 3),
#         tf.ones_like(x["flow_time_dist"]) * 2,
#         x["flow_time_dist"],
#     )
#     return x, y


# def fix_data(old_name, new_name):
#     num_partitions = len(os.listdir(old_name))
#     os.makedirs(new_name, exist_ok=True)
#     for ii in range(num_partitions):
#         ds = (
#             tf.data.Dataset.load(
#                 os.path.join(old_name, f"{ii:04d}"), compression="GZIP"
#             )
#             .prefetch(tf.data.experimental.AUTOTUNE)
#             .map(fix_time_dist)
#         )
#         ds.save(os.path.join(new_name, f"{ii:04d}"), compression="GZIP")

# fix_data("old_split_ds/training", "split_ds/training")
# fix_data("old_split_ds/validation", "split_ds/validation")
# fix_data("old_split_ds/test", "split_ds/test")
# fix_data("old_combined_ds/training", "combined_ds/training")
# fix_data("old_combined_ds/validation", "combined_ds/validation")
# fix_data("old_combined_ds/test", "combined_ds/test")