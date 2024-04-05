# prionn.py
import gensim  # word2vec model
import os
import pickle
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.metrics.regression.r2_score import R2Score
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch import argmax, squeeze, flatten
from torch.utils.data import TensorDataset, DataLoader

# Hyperparameter tuning
from ray import tune
from ray.air import session
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from cnn import CnnRegressor


VECTOR_SIZE = 4
BATCH_SIZE = 32
SCRIPT_LEN = 50
OUTPUT_SIZE = 1440


# Hyperparameter search space.
config = {
    'w2v_window': tune.choice([i for i in range(1, 10+1)]),
    'w2v_vec_size': tune.choice([i for i in range(1, 10+1)]),
    'w2v_epochs': tune.choice([i for i in range(1, 100+1)]),
    'script_len': tune.choice([10, 50, 100, 200]),
    'cnn_epochs': tune.choice([2 ** i for i in range(9)]),
    'cnn_batch_size': tune.choice([2 ** i for i in range(9)]),
    'cnn_lr': tune.loguniform(1e-4, 1e-1),
    'cnn_cl1': tune.choice([2 ** i for i in range(9)]),
    'cnn_cl2': tune.choice([2 ** i for i in range(9)]),
    'cnn_cl3': tune.choice([2 ** i for i in range(9)]),
    'cnn_f1': tune.choice([2 ** i for i in range(9)]),
    'cnn_f2': tune.choice([2 ** i for i in range(9)]),
    'cnn_f3': tune.choice([2 ** i for i in range(9)]),
}


class Test:
    def __init__(self):
        # Job script
        self.X = str()
        # Corresponding run time
        self.y = float()


class App:
    def __init__(self):
        self.tests = {}


class Data:
    def __init__(self):
        self.apps = {}


def get_data():
    """ Load testing data into X and y. """
    data = Data()
    pickle_path = "./data.pik"
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    # The location where tests are stored.
    TEST_DIR = "/Users/kenneth/Documents/OneDrive Bakup/Graduate School/Research Projects/Input-Based Prediction/ICPE/Miscellaneous/tests"
    # Enumerate the folders in TEST_DIR.
    apps = os.scandir(TEST_DIR)
    # NOTE: This approach is meant to generalize to all input scripts, so we no
    # longer create one model for each application.
    for app in apps:
        if not app.is_dir():
            continue
        # Skip metadata.
        if "PaxHeaders" in app.name:
            continue
        # Open the associated CSV with run times.
        dataset_csv = pd.read_csv(TEST_DIR + "/" + app.name + "dataset.csv")
        # Go through each test specified in the CSV.
        for _, row in dataset_csv.iterrows():
            # Get the index.
            index = row[0]
            if index % 100 == 0:
                print(index)
            # Get the path containing associated input files.
            test_path = Path(TEST_DIR + "/" + app.name +
                             "/" + str(index).zfill(10))
            # Open the associated test case.
            if app.name not in data.apps:
                data.apps[app.name] = App()
            if index not in data.apps[app.name].tests:
                data.apps[app.name].tests[index] = Test()
            case = data.apps[app.name].tests[index]
            try:
                with open(test_path / "submit.slurm", "r", encoding="utf-8") as text_file:
                    case.X = text_file.read()
            except FileNotFoundError as err:
                print(str(text_file.name)+" does not exist for test " +
                      str(index)+". Skipping. ", err)
                continue
            # Associate the time taken, in minutes.
            # DEBUG: In seconds, for now.
            case.y = float(row["timeTaken"]) #/60
    # Pickling the data essentially "caches" the data so we only have to read
    # the input files once.
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    return data

def list_to_loader(X, y):
    """ Convert a dataset from two matched lists into a loader. """

    # Convert datasets into PyTorch tensors using the TensorDataset wrapper
    X = torch.from_numpy(np.array(X))
    # Convert y to the probability of each run time.
    buckets = []
    for val in y:
        array = [0] * OUTPUT_SIZE
        minutes = int(min(val, OUTPUT_SIZE - 1))
        array[minutes] = 1
        buckets.append(array)
    # Convert y data to float to prevent a mismatch from the model predictions later.
    y = torch.from_numpy(np.array(buckets)).type(torch.float)
    data = TensorDataset(X, y)
    # Put the tensors into a loader.
    # NOTE: drop_last prevents a mismatch in batch size at the end of training.
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last = True)
    return loader

def model_loss(model, dataset, train = False, optimizer = None):
  # first calculated for the batches and at the end get the average
  performance = nn.MSELoss()
  # performance = nn.L1Loss()
  
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

    if(train):
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


def train_cnn(loader):
    pickle_path = "./model.pik"
    # Load an existing pickled model, if it exists.
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            return pickle.load(pickle_file)

    # The number of epochs to train for.
    NUM_EPOCHS = 25
    # Create the model.
    # TODO: Output buckets.
    model = CnnRegressor(BATCH_SIZE, SCRIPT_LEN*VECTOR_SIZE, OUTPUT_SIZE)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr = 0.007)

    # Train the model.
    for epoch in range(NUM_EPOCHS):
        avg_loss, avg_r2_score = model_loss(model, loader, train = True, optimizer = optimizer)
        print("Epoch " + str(epoch + 1) + ":\n\tLoss = " + str(avg_loss) + "\n\tR^2 Score = " + str(avg_r2_score))
    # Set the model to evaluation mode, as training has ended.
    model.eval()
    # Pickle the trained model.
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(model, pickle_file)
    # Return the trained model.
    return model


def predict(model, X, y):
    predictions = []
    # Load the data.
    loader = list_to_loader(X, y)
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


def main():
    data = get_data()
    # For now, we don't care about the specific app.
    # Group all tests together.
    X = [test.X for app in data.apps for test in data.apps[app].tests.values()]
    y = [test.y for app in data.apps for test in data.apps[app].tests.values()]

    # Preprocessing.
    # Split each document into a list of words.
    # These will be converted into vectors by word2vec.
    X = [doc.split() for doc in X]

    # Split into training and testing data.
    # NOTE: Since we are pickling the model, we want random_state fixed to
    # ensure the same split is used each time.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)

    word2vec_path = 'model/w2v.bin'
    # if os.path.isfile(word2vec_path):
    #     model = gensim.models.Word2Vec.load(word2vec_path)
    # else:

    # Add "[UNK]" token example to use when a token is missing.
    X_train.append(["[UNK]"])
    # Add a corresponding run time.
    y_train.append(0.0)

    # Generate the vector embeddings.
    documents = X_train
    model = gensim.models.Word2Vec(
        documents, vector_size=VECTOR_SIZE, window=2, min_count=0, workers=10, trim_rule=None)
    model.train(documents, total_examples=len(documents), epochs=10)
    # model.wv.save_word2vec_format(word2vec_path)

    # Get the vector values to pass into the CNN.
    def make_vector(X, model):
        X_vec = []
        for i, script in enumerate(X):
            sequence = []
            for j, word in enumerate(script):
                if word in model.wv:
                    sequence.append(model.wv[word])
                else:
                    print("Missing word: "+str(word) +
                          ", which is word #"+str(j)+" in script #"+str(i))
                    sequence.append(model.wv["[UNK]"])
            X_vec.append(sequence)
        return X_vec
    X_train_vec = make_vector(X_train, model)
    X_test_vec = make_vector(X_test, model)

    # Pad/truncate all scripts to the same length.
    def pad_to_len(script, target_len):
        script_len = len(script)
        pad_len = target_len-script_len
        if script_len < target_len:
            print("Adding " + str(pad_len) +
                  " words to script of length " + str(script_len))
            for _ in range(pad_len):
                script.append(model.wv["[UNK]"])
        if script_len > target_len:
            to_strip = abs(pad_len)
            print("Stripping " + str(to_strip) +
                  " words from script of length " + str(script_len))
            for _ in range(to_strip):
                script.pop()
    # TODO: Increase this later, but keep it short for testing.
    target_len = SCRIPT_LEN
    for script in X_train_vec:
        pad_to_len(script, target_len)
    for script in X_test_vec:
        pad_to_len(script, target_len)

    # Convert training data to a loader.
    loader_train = list_to_loader(X_train_vec, y_train)
    model = train_cnn(loader_train)
    predictions = predict(model, X_test_vec, y_test)

    for predicted, actual in predictions:
        print(f"Prediction: {predicted}\tActual: {actual}")
    return


if __name__ == "__main__":
    # Run main.
    main()


# High-level pseudocode to make sense of everything:
# Train word2vec on existing scripts.
# vector embeddings = word2vec("data/input")
# Pass embeddings as inputs to cnn.
# model = train_2dcnn(vector embeddings, run times)
# Make predictions on new inputs.
# prediction = model(script)
