# data_loader.py
import gensim  # word2vec model
import os
import pickle
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import params


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
    pickle_path = "C:\\Users\\MarioMan\\Documents\\Resilio Sync\\prionn-plus\\pickles\\data.pik" # Windows
    # pickle_path = "/mnt/c/Users/MarioMan/Documents/Resilio Sync/prionn-plus/pickles/data.pik" # WSL2
    # pickle_path = "pickles/data.pik"
    if os.path.isfile(pickle_path):
        with open(pickle_path, "rb") as pickle_file:
            print("Found existing pickle file for data_loader!")
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
            case.y = float(row["timeTaken"])  # /60
    # Pickling the data essentially "caches" the data so we only have to read
    # the input files once.
    with open(pickle_path, "wb") as pickle_file:
        pickle.dump(data, pickle_file)
    return data


def list_to_loader(config, X, y):
    """ Convert a dataset from two matched lists into a loader. """

    # Convert datasets into PyTorch tensors using the TensorDataset wrapper
    X = torch.from_numpy(np.array(X))
    # Convert y to the probability of each run time.
    buckets = []
    for val in y:
        array = [0] * config['output_size']
        minutes = int(min(val, config['output_size'] - 1))
        array[minutes] = 1
        buckets.append(array)
    # Convert y data to float to prevent a mismatch from the model predictions later.
    y = torch.from_numpy(np.array(buckets)).type(torch.float)
    data = TensorDataset(X, y)
    # Put the tensors into a loader.
    # NOTE: drop_last prevents a mismatch in batch size at the end of training.
    loader = DataLoader(data, batch_size=config['cnn_batch_size'],
                        shuffle=True, drop_last=True)
    return loader


def process_data(config, data=None):
    if data is None:
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

    # Add "[UNK]" token example to use when a token is missing.
    X_train.append(["[UNK]"])
    # Add a corresponding run time.
    y_train.append(0.0)

    # Generate the vector embeddings.
    documents = X_train
    model = gensim.models.Word2Vec(
        documents, vector_size=config['w2v_vec_size'], window=config['w2v_window'], min_count=0, workers=params.CPU_COUNT, trim_rule=None)
    model.train(documents, total_examples=len(documents),
                epochs=config['w2v_epochs'])

    # Get the vector values to pass into the NN.
    def make_vector(X, model):
        X_vec = []
        for i, script in enumerate(X):
            sequence = []
            for j, word in enumerate(script):
                if word in model.wv:
                    sequence.append(model.wv[word])
                else:
                    # print("Missing word: "+str(word) +
                    #       ", which is word #"+str(j)+" in script #"+str(i))
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
            # print("Adding " + str(pad_len) +
            #       " words to script of length " + str(script_len))
            for _ in range(pad_len):
                script.append(model.wv["[UNK]"])
        if script_len > target_len:
            to_strip = abs(pad_len)
            # print("Stripping " + str(to_strip) +
            #       " words from script of length " + str(script_len))
            for _ in range(to_strip):
                script.pop()
    target_len = config['script_len']
    for script in X_train_vec:
        pad_to_len(script, target_len)
    for script in X_test_vec:
        pad_to_len(script, target_len)

    # Convert training data to a loader.
    loader_train = list_to_loader(config, X_train_vec, y_train)
    loader_test = list_to_loader(config, X_test_vec, y_test)

    return (loader_train, loader_test)
