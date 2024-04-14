# tisc: pytorch-time-series-classification

[![PyPI version](https://badge.fury.io/py/tisc.svg)](https://badge.fury.io/py/tisc)
[![Downloads](https://pepy.tech/badge/tisc)](https://pepy.tech/project/tisc)
[![Downloads](https://pepy.tech/badge/tisc/month)](https://pepy.tech/project/tisc)
[![Downloads](https://pepy.tech/badge/tisc/week)](https://pepy.tech/project/tisc)

Simple model creation and training framework for <b>ti</b>me <b>s</b>eries <b>c</b>lassification in Pytorch.

## What can you do with tisc?

<b>`tisc` is a simple framework for time series classification in Pytorch.</b>

- You can create a Pytorch model for time series classification with just one function.
- You can choose the model from many supported models.
- You can train the model with just one method.
- You can evaluate or predict with the trained model with just one method.

## Setup

### 1. Install tisc
```bash
pip install tisc
```

### 2. Install Pytorch

If Pytorch is not installed to your environment, you have to install Pytorch that matches your environment from the official website: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

example (this command is for my environment):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### 0. Prepare the dataset / dataloader

#### Time series data

The time series data should be a 3D tensor with the shape of `(number_of_samples, timestep, dimentions)`.

For example, if you have a dataset with 1000 samples, each sample has 20 timesteps, and each timestep has 100 dimentions, the shape of the dataset should be `(1000, 20, 100)`.

The label should be a 1D tensor with the shape of `(number_of_samples,)`.

For example, if you have a dataset with 1000 samples, the shape of the label should be `(1000,)`.

```python
import torch

# The shape of the time series data should be (number_of_samples, timestep, dimentions)
print(train_data.shape)  # (1000, 20, 100)

# The shape of the label should be (number_of_samples,)
print(train_label.shape)  # (1000,)
```

#### Dataset

`tisc` supports the dataset that is a subclass of `torch.utils.data.Dataset`. 

The dataset should return a tuple of `(data, label)` in the `__getitem__` method.

You can use `TensorDataset` from `torch.utils.data` to create a dataset from the time series data and the label easily.

```python
import torch
from torch.utils.data import TensorDataset

# Prepare the dataset with TensorDataset
train_dataset = TensorDataset(train_data, train_label)
val_dataset = TensorDataset(val_data, val_label)
test_dataset = TensorDataset(test_data, test_label)

# Check the type of the dataset
print(type(train_dataset))  # <class 'torch.utils.data.dataset.TensorDataset'>
```

#### Dataloader

You have to use `torch.utils.data.DataLoader` to load the dataset.

```python
import torch
from torch.utils.data import DataLoader

# Prepare the dataset
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Check the type of the dataloader
print(type(train_loader))  # <class 'torch.utils.data.dataloader.DataLoader'>
```
### 1. Create a classifier

You can create a classifier with the `build_classifier` function.

The `build_classifier` function returns a `tisc.Classifier` object.

A `Classifier` object contains the model, the optimizer, the loss function, and the training and evaluation methods.

When you create a classifier, you have to pass the following arguments:

- `model_name`: The name of the model. The model should be one of the supported models. (e.g., `'LSTM'`, `'BiLSTM'`, `'Transformer'`)
- `timestep`: The number of timesteps in the time series data.
- `dimentions`: The number of dimentions in each timestep.
- `num_classes`: The number of classes in the dataset.

```python
import tisc

# Create a classifier
classifier = tisc.build_classifier(model_name='LSTM',
                                   timestep=20,
                                   dimentions=100,
                                   num_classes=10)

# Check the type of the classifier
print(type(classifier))  # <class 'tisc.Classifier'>
```

### 2. Train the classifier

You can train the classifier with the `train` method.

The `train` method requires the following arguments:

- `epochs`: The number of epochs to train the classifier.
- `train_loader`: The dataloader for the training dataset.

you can pass `val_loader` to train the classifier with validation.

```python
classifier.train(train_loader, epochs=100)

# If the `val_loader` argument is passed, you can train the classifier with validation.
classifier.train(train_loader, val_loader=val_loader, epochs=100)
```

### 3. Evaluate the classifier

You can evaluate the classifier with the `evaluate` method.

The `evaluate` method requires the following arguments:

- `test_loader`: The dataloader for the test dataset.

The `evaluate` method can return the classification report and the confusion matrix if you pass the `return_report` and `return_confusion_matrix` arguments as `True`.

If `with_best_model` argument is `True`, the classifier will use the best model that marked the best result about the model saving strategy. 

```python
classifier.evaluate(test_loader,
                    return_report=True,
                    return_confusion_matrix=True,
                    with_best_model=True)
```

## Supported models

The models that can be used in version 0.1.0:

- LSTM
- BiLSTM
- Transformer

and more! (More models will be added.)

