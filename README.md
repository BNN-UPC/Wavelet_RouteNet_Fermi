# Wavelet-Enhanced Graph Neural Networks: Towards Non-Parametric Network Traffic Modeling

**Carlos Güemes Palau, Miquel Ferrior Galmés, Jordi Paillisse Vilanova, Albert López Brescó, Pere Barlet Ros, Albert Cabellos Aparicio**

This repository is the code of the paper *Wavelet-Enhanced Graph Neural Networks: Towards Non-Parametric Network Traffic Modeling* (publication pending)

Contact us: *[carlos.guemes@upc.edu](mailto:carlos.guemes@upc.edu)*, *[contactus@bnn.upc.edu](mailto:contactus@bnn.upc.edu)*

## Abstract

Network modeling is crucial for the design, management, and optimization of modern telecommunications and data networks. Recent advancements in Machine Learning (ML), specifically Graph Neural Networks (GNNs), offer promising solutions but rely on parameterized traffic representations, necessitating retraining for new traffic patterns. This paper explores integrating the Discrete Wavelet Transform (DWT) with GNNs to enhance network traffic modeling. By leveraging wavelets, which decompose signals into both time and frequency components, we aim to encode traffic patterns without assuming specific distributions, improving model adaptability and accuracy. We modify the state-of-the-art RouteNet-Fermi model to incorporate wavelet-based traffic encoding and evaluate its performance across different synthetic and real traffic scenarios. Our findings show that wavelet-based encoding handles unseen traffic distributions with minimal impact on performance, unlike traditional parameter-based approaches. This work represents a step forward in bridging the gap between non-parametric traffic representation and advanced network modeling, offering a promising solution for dynamic and complex network environments.

# Quickstart

Please ensure that your OS

1. Please ensure that your OS has installed Python 3 (ideally 3.10)
2. Create the virtual environment and activate the environment:
```bash
virtualenv -p python3 myenv
source myenv/bin/activate
```
3. Then we install the required packages (to avoid issues, make sure to install the specific packages versions, especially for tensorflow):
```bash
pip install tensorflow==2.15.0 numpy==1.26.3 matplotlib==3.8.2 notebook==7.0.7
```

Once those are ready you can:
- Train your own models [`train_baseline.py`](train_baseline.py) and [`train_wavelet.py`](train_wavelet.py).
- Evaluate the the trained models [`evaluation.ipynb`].(evaluation.ipynb)

# Repository structure

The repository contains the following structure:
- `ckpt`: Folder containing the checkpoints used in the paper evaluation.
- `data`: Folder containing the data used in the project:
  - `combined_ds`: Dataset in which all the traffic distributions are present across all partitions
  - `split_ds`: Dataset in which parametric distributions were used only for training and validation; the test partition is solely formed by samples from an unseen non-parametric distribution.
- [`train_baseline.py`](train_baseline.py): script to train RouteNet-Fermi with its original approach (baseline).
- [`train_wavelet.py`](train_wavelet.py): script to train RouteNet-Fermi with the proposed wavelet-decomposition approach.
- [`evaluation.ipynb`](evaluation.ipynb): notebook folder to evaluate the trained models.
- [`utils.py`](utils.py) and [`data/data_loader.py`](data/data_loader.py) contains auxiliary functions for reading the datasets and training
- [LICENSE](LICENSE): see the file for the full license.

# Modifying the training scripts

Both the [`train_baseline.py`](train_baseline.py) and [`train_wavelet.py`](train_wavelet.py) scripts contain the default configuration used. Follow comments in the code to perform your own modifications. Here is a quick reference guide.
- Use the `RUN_EAGERLY` variable (line 19) to run TensorFlow in eager mode.
- Use the `RELOAD_WEIGHTS` variable (line 22) to resume training from a specific checkpoint.
- Select the dataset to be used to run `ds_path` (line 25) .
- Modify the experiments identifier (and the path on the results) by modifying variables `batch_name` and `experiment_name` (lines 28-38).
- (Only [`train_wavelet.py`](train_wavelet.py)): select the model to use by defining `model_class` (line 29).
- Modify `MODEL_PARAMS` to define its hyperparameters (line 56).
- Training parameters are found in lines 54-57.

# License

See the [file](LICENSE) for the full license:


```
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
```
