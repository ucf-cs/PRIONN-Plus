# transformer.py
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


class TransformerRegressor(nn.Module):
    def __init__(self, config):
        super(TransformerRegressor, self).__init__()
        self.transformer = nn.Transformer(config['d_model'], config['nhead'], config['num_layers'])
        self.regressor_head = nn.Linear(config['d_model'], config['num_targets'])
        
    def forward(self, config, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        src = src.view(config['batch_size'], -1, config['d_model'])
        tgt = tgt.view(config['batch_size'], -1, config['d_model'])

        # Pass the source sequence through the transformer
        memory = self.transformer(src, tgt, src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        
        # Use the final state of the transformer as input to the regressor head
        output = self.regressor_head(memory)
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
