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

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from random import seed
import numpy as np
from models import RouteNet_Fermi
from utils import log_transform, get_fermi_z_scores, denorm_mape_metric
from data.data_loader import load_partition


# Set all seeds
SEED = 1
seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# RUN EAGERLY -> True for debugging
RUN_EAGERLY = False
tf.config.run_functions_eagerly(RUN_EAGERLY)
# RELOAD_WEIGHTS -> True to continue training from a checkpoint
RELOAD_WEIGHTS = False

# SELECT DS:
ds_path = "data/splits_ds"

# EXPERIMENT NAME AND PATHS
z_score_path = "data/fermi_normalization/original/splits_ds/z_scores.pkl"
batch_name = "revisions"
experiment_name = None  # Used to distinguish smaller differences
experiment_path = (
    f"{batch_name}/{RouteNet_Fermi.name}"
    if experiment_name in [None, ""]
    else f"{batch_name}/{experiment_name}/{RouteNet_Fermi.name}"
)
os.makedirs(f"ckpt/{experiment_path}", exist_ok=True)
os.makedirs(f"tensorboard/{experiment_path}", exist_ok=True)
os.makedirs(os.path.dirname(z_score_path), exist_ok=True)

ds_train = (
    load_partition(os.path.join(ds_path, "training"))
    .map(log_transform)
    .prefetch(tf.data.experimental.AUTOTUNE)
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
)
ds_val = (
    load_partition(os.path.join(ds_path, "validation"))
    .map(log_transform)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# MODEL PARAMETERS
EPOCHS = 300
STEPS_PER_EPOCH = 500
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
loss = tf.keras.losses.MeanSquaredError()
MODEL_PARAMS = {
    "log": True,  # Used to that log-mse is used as loss function
    "max_num_dist": 2,  # Should be equal to the number of distributions seen in training
    "z_scores": get_fermi_z_scores(
        ds_train,
        RouteNet_Fermi.z_scores_fields,
        summarize=1000,
        flatten=True,
        store_res_path=z_score_path,
        check_existing=True,
    ),
}
# Build and compile model
model = RouteNet_Fermi(**MODEL_PARAMS)
model.compile(
    optimizer=optimizer,
    loss=loss,
    run_eagerly=RUN_EAGERLY,
    metrics=[denorm_mape_metric],
)

# Load checkpoint if available
ckpt_dir = f"ckpt/{experiment_path}"
latest = tf.train.latest_checkpoint(ckpt_dir)
if RELOAD_WEIGHTS and latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.4f}")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    save_best_only=False,
    save_weights_only=True,
    save_freq="epoch",
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f"tensorboard/{experiment_path}", histogram_freq=1
)
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    factor=0.5, patience=10, verbose=1, cooldown=3, mode="min", monitor="loss"
)

model.fit(
    ds_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=ds_val,
    callbacks=[
        cp_callback,
        tensorboard_callback,
        reduce_lr_callback,
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    use_multiprocessing=True,
)
