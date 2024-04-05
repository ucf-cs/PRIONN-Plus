# hyper_param.py
import os
import torch
import torch.nn as nn
from functools import partial
from ray import tune
from ray.tune.schedulers import ASHAScheduler

import cnn
import data_loader
import params


def test_accuracy(model, testloader, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# TODO: Bump this up to test more combinations of parameters.
num_samples = 10000

max_num_epochs = 10
gpus_per_trial = torch.cuda.device_count()/params.CPU_COUNT
# Hyperparameter search space.
config = {
    'w2v_vec_size': tune.choice([i for i in range(1, 10+1)]),  # 4
    'w2v_window': tune.choice([i for i in range(1, 10+1)]),  # 2
    'w2v_epochs': tune.choice([i for i in range(1, 100+1)]),  # 10
    'script_len': tune.choice([10, 50, 100, 200]),  # 50
    'cnn_epochs': tune.choice([2 ** i for i in range(7)]),
    'cnn_batch_size': tune.choice([2 ** i for i in range(9)]),  # 32
    'cnn_lr': tune.loguniform(1e-4, 1e-1),
    'cnn_cl1': tune.choice([2 ** i for i in range(9)]),
    'cnn_cl2': tune.choice([2 ** i for i in range(9)]),
    'cnn_cl3': tune.choice([2 ** i for i in range(9)]),
    'cnn_f1': tune.choice([2 ** i for i in range(9)]),
    'cnn_f2': tune.choice([2 ** i for i in range(9)]),
    'cnn_f3': tune.choice([2 ** i for i in range(9)]),
    'output_size': 1440
}
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2,
)
result = tune.run(
    partial(cnn.train_get_metrics),
    resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    # checkpoint_at_end=True)
    checkpoint_at_end=False,
    resume=True) # Resume existing trials.

best_trial = result.get_best_trial("loss", "min", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
print(
    f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

# best_trained_model = cnn.CnnRegressor(best_trial.config)
# device = "cpu"
# if torch.cuda.is_available():
#     device = "cuda:0"
#     if gpus_per_trial > 1:
#         best_trained_model = nn.DataParallel(best_trained_model)
# best_trained_model.to(device)

# best_checkpoint = best_trial.checkpoint
# best_checkpoint_data = best_checkpoint.to_dict()

# best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

# test_acc = test_accuracy(best_trained_model, device)
# print("Best trial test set accuracy: {}".format(test_acc))
