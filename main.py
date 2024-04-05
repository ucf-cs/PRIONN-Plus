# main.py
import torch
import torch.nn as nn

from functools import partial
from ray import tune

import data_loader
import cnn
import params


def main():
    (loader_train, loader_test) = data_loader.process_data(params.config)

    device = "cpu" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = cnn.create_model(params.config)

    # Wrap the model with DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # Move the model to the device
    model.to(device)

    if torch.cuda.device_count() > 1:
        for _, data in enumerate(loader_train, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    model, metrics = cnn.train_cnn(params.config, model, loader_train)

    if torch.cuda.device_count() > 1:
        for _, data in enumerate(loader_test, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    predictions = cnn.predict(model, loader_test)

    for predicted, actual in predictions:
        print(f"Prediction: {predicted}\tActual: {actual}")
    return


if __name__ == "__main__":
    # Run main.
    main()
