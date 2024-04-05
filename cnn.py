# cnn.py
import torch.nn as nn
import torch.nn.functional as F

import os
import pickle
import tempfile
import torch
import torch.optim as optim
from ignite.contrib.metrics.regression.r2_score import R2Score
from ray import train, tune
from ray.train import Checkpoint
from ray.train.context import TrainContext

from torch import flatten
from torch.utils.data import random_split

import data_loader
import params


class CnnRegressor(nn.Module):
    """ Define the CNN model. """

    def __init__(self, config):
        # initialization of the superclass
        super(CnnRegressor, self).__init__()

        # store the parameters
        self.batch_size = config['cnn_batch_size']
        self.inputs = config['script_len'] * config['w2v_vec_size']
        self.outputs = config['output_size']
        # DEBUG:
        # print(f"CNNRegressor init with these parameters:",
        #       f"batch_size: {self.batch_size}",
        #       f"inputs: {self.inputs}",
        #       f"outputs: {self.outputs}")

        # define the input layer
        self.input_layer = nn.Conv1d(self.inputs, self.batch_size, 1, stride=1)

        # define max pooling layer
        self.max_pooling_layer = nn.MaxPool1d(1)

        # define other convolutional layers
        self.conv_layer1 = nn.Conv1d(
            self.batch_size, config['cnn_cl1'], 1, stride=3)
        self.conv_layer2 = nn.Conv1d(
            config['cnn_cl1'], config['cnn_cl2'], 1, stride=3)
        self.conv_layer3 = nn.Conv1d(
            config['cnn_cl2'], config['cnn_cl3'], 1, stride=3)

        # define the flatten layer
        self.flatten_layer = nn.Flatten()

        # Define the fully connected layers
        self.full1 = nn.Linear(config['cnn_cl3'],  config['cnn_f1'])
        self.full2 = nn.Linear(config['cnn_f1'],  config['cnn_f2'])
        self.full3 = nn.Linear(config['cnn_f2'],  config['cnn_f3'])

        # define the output layer
        self.output_layer = nn.Linear(config['cnn_f3'], config['output_size'])

    # define the method to feed the inputs to the model
    def forward(self, input):
        # DEBUG: 
        # print(input.shape, self.batch_size, self.inputs)

        # input is reshaped to the 1D array and fed into the input layer
        new_shape = (self.batch_size, self.inputs, 1)

        # Calculate the total number of elements in the input tensor
        total_elements_input = torch.prod(torch.tensor(input.shape)).item()
        # Calculate the total number of elements required by the new shape
        total_elements_new_shape = torch.prod(torch.tensor(new_shape)).item()
        # Check if the reshaping is possible
        if total_elements_input != total_elements_new_shape:
            # breakpoint()
            print(f"Reshaping failed: input tensor has {total_elements_input} elements, but the new shape requires {total_elements_new_shape} elements.")
            return None
        
        # If reshaping is possible, perform the reshape
        input = input.reshape(new_shape)

        # ReLU is applied on the output of input layer
        output = F.relu(self.input_layer(input))

        # max pooling is applied and then Convolutions are done with ReLU
        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer1(output))

        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer2(output))

        output = self.max_pooling_layer(output)
        output = F.relu(self.conv_layer3(output))

        # flatten layer is applied
        output = self.flatten_layer(output)

        # linear layer and ReLu is applied
        output = F.relu(self.full1(output))
        output = F.relu(self.full2(output))
        output = F.relu(self.full3(output))

        # finally, output layer is applied
        output = self.output_layer(output)
        return output


def model_loss(model, dataset, config, train=False, optimizer=None):
    # first calculated for the batches and at the end get the average
    performance = nn.CrossEntropyLoss()

    score_metric = R2Score()

    avg_loss = 0
    avg_score = 0
    count = 0

    for input, output in iter(dataset):
        # get predictions of the model for training set
        predictions = model(input)

        # Remove the extra dimensions
        predictions = flatten(predictions)
        output = flatten(output)

        # calculate loss of the model
        loss = performance(predictions, output)
        # compute the R2 score
        score_metric.update([predictions, output])
        score = score_metric.compute()

        if (train):
            # clear the errors
            optimizer.zero_grad()
            # compute the gradients for optimizer
            loss.backward()
            # use optimizer in order to update parameters
            # of the model based on gradients
            optimizer.step()

        # store the loss and update values
        avg_loss += loss.item()
        avg_score += score
        count += 1

    return avg_loss/count, avg_score/count

def create_model(config):
    return CnnRegressor(config)

def train_get_metrics(config, model=None, loader=None):
    model, metrics = train_cnn(config=config, model=model, loader=loader)
    return metrics


def train_cnn(config, model=None, loader=None):
    if model is None:
        model = create_model(config)
    if loader is None:
        loader = data_loader.process_data(config)[0]

    # DEBUG: GPU isn't working right now, as some data seems to be on the CPU. We just disable the functionality for now.
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # Wrap the model with DataParallel if multiple GPUs are available
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    # # Move the model to the device
    # model.to(device)
    # if torch.cuda.device_count() > 1:
    #     for _, data in enumerate(loader, 0):
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)

    pickle_path = "./model.pik"
    # DEBUG: Disabled for now.
    # # Load an existing pickled model, if it exists.
    # if os.path.isfile(pickle_path):
    #     with open(pickle_path, "rb") as pickle_file:
    #         return pickle.load(pickle_file)

    # The number of epochs to train for.
    epochs = config['cnn_epochs']
    # optimizer = optim.SGD(model.parameters(), lr=config['cnn_lr'], momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=config['cnn_lr'])

    # DEBUG: Disable checkpoints for now.
    checkpoint = None # train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_dict = torch.load(
                os.path.join(checkpoint_dir, "checkpoint.pt"))
            start_epoch = checkpoint_dict["epoch"] + 1
            model.load_state_dict(checkpoint_dict["model_state"])
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    else:
        start_epoch = 0

    # Train the model.
    for epoch in range(start_epoch, epochs):
        avg_loss, avg_r2_score = model_loss(
            model, loader, config, train=True, optimizer=optimizer)
        print("Epoch " + str(epoch + 1) + ":\n\tLoss = " +
              str(avg_loss) + "\n\tR^2 Score = " + str(avg_r2_score))
        # DEBUG: Disable checkpoints for now.
        # # TODO: Are checkpoints useful to us?
        # with tempfile.TemporaryDirectory() as tempdir:
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "net_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #         },
        #         os.path.join(tempdir, "checkpoint.pt"),
        #     )
        #     train.report(metrics={"loss": avg_loss, "accuracy": avg_r2_score},
        #                  checkpoint=Checkpoint.from_directory(tempdir))
    
     # After training and evaluation, report the final metrics and checkpoint
    with tempfile.TemporaryDirectory() as tempdir:
        torch.save(
            {
                "epoch": epochs - 1, # Assuming epochs is the total number of epochs
                "net_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            os.path.join(tempdir, "checkpoint.pt"),
        )
        # Assuming avg_loss and avg_r2_score are the final metrics you want to report
        print(f"Saving loss: {avg_loss}, accuracy: {avg_r2_score}, and checkpoint {Checkpoint.from_directory(tempdir)} from {tempdir}")
        metrics={"loss": avg_loss, "accuracy": avg_r2_score}
        train.report(metrics,
                     checkpoint=Checkpoint.from_directory(tempdir))

    # Set the model to evaluation mode, as training has ended.
    model.eval()
    # Pickle the trained model.
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    # Return the trained model.
    return (model, metrics)

    # model = CnnRegressor(config)
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if torch.cuda.device_count() > 1:
    #         model = nn.DataParallel(model)
    # model.to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=config['cnn_lr'], momentum=0.9)
    # # optimizer = optim.Adam(model.parameters(), lr=config['cnn_lr'])

    # checkpoint = train.get_checkpoint()
    # if checkpoint:
    #     with checkpoint.as_directory() as checkpoint_dir:
    #         checkpoint_dict = torch.load(
    #             os.path.join(checkpoint_dir, "checkpoint.pt"))
    #         start_epoch = checkpoint_dict["epoch"] + 1
    #         model.load_state_dict(checkpoint_dict["model_state"])
    #         optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
    # else:
    #     start_epoch = 0

    # # TODO: Not sure why we're splitting our already-split data again here.
    # test_abs = int(len(loader) * 0.8)
    # train_subset, val_subset = random_split(
    #     loader, [test_abs, len(loader) - test_abs]
    # )

    # train_loader = torch.utils.data.DataLoader(
    #     train_subset, batch_size=int(config["cnn_batch_size"]), shuffle=True, num_workers=params.CPU_COUNT
    # )
    # val_loader = torch.utils.data.DataLoader(
    #     val_subset, batch_size=int(config["cnn_batch_size"]), shuffle=True, num_workers=params.CPU_COUNT
    # )

    # # loop over the dataset multiple times
    # for epoch in range(start_epoch, config['cnn_epochs']):
    #     running_loss = 0.0
    #     epoch_steps = 0
    #     for i, data in enumerate(train_loader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)

    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()

    #         # print statistics
    #         running_loss += loss.item()
    #         epoch_steps += 1
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print(
    #                 "[%d, %5d] loss: %.3f"
    #                 % (epoch + 1, i + 1, running_loss / epoch_steps)
    #             )
    #             running_loss = 0.0

    #     # Validation loss
    #     val_loss = 0.0
    #     val_steps = 0
    #     total = 0
    #     correct = 0
    #     for i, data in enumerate(val_loader, 0):
    #         with torch.no_grad():
    #             inputs, labels = data
    #             inputs, labels = inputs.to(device), labels.to(device)

    #             outputs = model(inputs)
    #             _, predicted = torch.max(outputs.data, 1)
    #             total += labels.size(0)
    #             correct += (predicted == labels).sum().item()

    #             loss = criterion(outputs, labels)
    #             val_loss += loss.cpu().numpy()
    #             val_steps += 1

    #     with tempfile.TemporaryDirectory() as tempdir:
    #         torch.save(
    #             {
    #                 "epoch": epoch,
    #                 "net_state_dict": model.state_dict(),
    #                 "optimizer_state_dict": optimizer.state_dict(),
    #             },
    #             os.path.join(tempdir, "checkpoint.pt"),
    #         )
    #         train.report(metrics={"loss": val_loss / val_steps, "accuracy": correct /
    #                      total}, checkpoint=Checkpoint.from_directory(tempdir))
    # print("Finished Training")
    # # Set the model to evaluation mode, as training has ended.
    # model.eval()
    # # Return the trained model.
    # return model


def predict(model, loader):
    predictions = []
    with torch.no_grad():
        for inputs, actual_buckets in loader:
            outputs = model(inputs)
            # Return the result with the highest probability for each test.
            # TODO: This does not at all work as I expected. This needs a rework to get the bucket with the highest probability and its associated runtime.
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(actual_buckets, 1)
            predictions.append((predicted, actual))
            print(f"Prediction: {predicted}\tActual: {actual}")
    return predictions
