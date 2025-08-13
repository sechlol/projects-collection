from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from metrics_collector import MetricsCollector

"""
This class provides a framework for training, testing, and predicting with a neural network model.
through the class "Gym", which encapsulates a machine learning training process. 
The class takes several parameters upon initialization, including a neural network model, a loss function, 
an optimizer, a metrics collector, command-line arguments, and a device to perform the computations on.

The class contains several methods for performing training, validation, testing, and prediction. 
The "_training_step" method performs a single training iteration for one epoch, where it loops through 
the training data and performs forward and backward passes through the network to update the weights. 
The "_validation_step" method evaluates the model's performance on the validation or test data by making 
predictions on the input data and calculating the loss and metrics.

The "train_model" method performs the entire training process by looping through the specified number of 
epochs and calling the "_training_step" and "_validation_step" methods at each epoch. It also prints out 
the validation metrics after each epoch. The "test_model" method evaluates the model's performance on the 
test data using the "_validation_step" method. Finally, the "predict" method takes a data loader as input 
and makes predictions on the data using the trained model.

Throughout the class, the "MetricsCollector" object is used to keep track of the metrics during training and 
validation, and the "torch" library is used for performing the computations on the specified device. 
"""


class Gym:
    def __init__(self,
                 model: nn.Module,
                 loss_function: Optional[nn.Module],
                 optimizer: Optional[torch.optim.Optimizer],
                 metrics: MetricsCollector,
                 cmd_args: Dict[str, Any],
                 device: str):

        self.model = model
        self.device = device
        self.metrics = metrics
        self.cmd_args = cmd_args
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.verbose = cmd_args.get("-v")

        # Command line arguments
        self.epochs = cmd_args.get("-e")

    def _training_step(self, epoch: int, train_loader: DataLoader):
        total_batches = (len(train_loader) + 1)

        self.model.train()      
        for i, (img, labels, img_id) in enumerate(train_loader):
            x, y = img.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()

            self.metrics.add_train_batch(epoch=epoch, batch_i=i, loss=loss.item(), y_pred=y_pred.detach().cpu(), y_true=labels)

            if self.verbose and i % ((total_batches // 10) + 1) == 0:
                train_epoch_report = self.metrics.get_train_epoch_report(epoch)
                print(f"\nEpoch {epoch}, batch {i}/{total_batches}.")
                print(train_epoch_report[["accuracy", "loss", "macro_f1", "micro_f1", "macro_precision",
                                          "micro_precision", "macro_recall", "micro_recall"]])

    def _validation_step(self, epoch: int, loader: DataLoader, is_test: bool = False):
        total_batches = (len(loader) + 1)

        # set the model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            for i, (img, labels, img_ids) in enumerate(loader):
                x, y = img.to(self.device), labels.to(self.device)

                # make the predictions and calculate the validation loss
                y_pred = self.model(x)

                y_pred_cpu = y_pred.cpu()
                if is_test:
                    self.metrics.add_test_batch(batch_i=i, y_pred=y_pred_cpu, y_true=labels)
                    self.metrics.add_prediction(img_ids=img_ids, y_pred=y_pred_cpu)
                else:
                    loss = self.loss_function(y_pred, y).item()
                    self.metrics.add_validation_batch(epoch=epoch, batch_i=i, loss=loss, y_pred=y_pred_cpu, y_true=labels)

                if self.verbose and i % (round(total_batches // 100) + 1) == 0:
                    print("Evaluation progress: {:.2f}%".format((i / total_batches) * 100))

    def train_model(self, train_loader: DataLoader, dev_loader: DataLoader):
        history = []
        for epoch in range(self.epochs):
            print(f"\n\n--- Beginning Epoch {epoch} ---")
            self._training_step(epoch, train_loader)

            print("\n\nValidating...")
            self._validation_step(epoch, dev_loader)
            val_epoch_report = self.metrics.get_validation_epoch_report(epoch)
            print(val_epoch_report[["accuracy", "loss", "macro_f1", "micro_f1", "macro_precision", "micro_precision",
                                    "macro_recall", "micro_recall"]])

        print(f"-- END TRAINING --")
        return self.model, history

    def test_model(self, test_loader: DataLoader):
        self._validation_step(epoch=0, loader=test_loader, is_test=True)

    def predict(self, data_loader: DataLoader):
        self.model.eval()
        with torch.no_grad():
            for i, (img, img_ids) in enumerate(data_loader):
                x = img.to(self.device)
                y_pred = self.model(x)
                self.metrics.add_prediction(img_ids=img_ids, y_pred=y_pred.cpu())
