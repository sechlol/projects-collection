import torch
import pandas as pd
import sklearn.metrics as skm

from typing import List, Any, Dict


class Phase:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    PREDICTION = "prediction"


# Phases in a list, for easier manipulation
_PHASES = [Phase.TRAIN, Phase.VAL, Phase.TEST]

# Metrics collected at each batch, they will be averaged over each epoch
_METRICS = ["loss", "micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"]

# Used to keep track of data points for aggregation and grouping
_INDICES = ["epoch", "batch_index", "batch_size"]

# Columns in the history dataframe of each phase
_COLUMNS = _INDICES + _METRICS

"""
The class is intended to be used in conjunction with a deep neural network model to collect accuracy and other
metrics at each step of the training process. It provides a convenient and flexible way to collect and analyze metrics, 
accuracy, and predictions during the training, validation, and testing phases.

It has methods for adding training, validation, test, and prediction batches to collect metrics at each step. 
It also provides methods for getting epoch-wise and batch-wise reports and histories of the collected metrics.

The class has an initializer method that takes a list of all possible textual labels for the data
and command-line arguments. The labels are used to collect accuracy score for each individual label, 
and the command-line arguments give the threshold when an output should be interpreted as "detected". 
The class uses pandas DataFrame to store and manipulate the collected data.
"""


class MetricsCollector:
    def __init__(self, labels: List[str], cmd_args):
        # Labels used for accuracy calculation
        self.labels = labels
        self.accuracy_columns = ["correct"] + labels
        self._all_columns = _COLUMNS + self.accuracy_columns
        self._correct_threshold = cmd_args.get("-t", 0.5)

        # Keep track of history for each batch, for each phase
        self._batch_history = {p: [] for p in _PHASES}
        self._history = {p: pd.DataFrame(columns=_COLUMNS) for p in _PHASES}
        self._predictions = pd.DataFrame(columns=["Filename"] + labels, dtype=int)
        self._test_predictions = {"pred": [], "true": []}

    @property
    def has_train_history(self) -> bool:
        """
        Returns true if the training history is not empty.
        """
        return not self._history[Phase.TRAIN].empty or bool(self._batch_history[Phase.TRAIN])

    def add_train_batch(self, epoch: int, batch_i: int, loss: float, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._add_batch_history(Phase.TRAIN, epoch, batch_i, loss, y_pred, y_true)

    def add_validation_batch(self, epoch: int, batch_i: int, loss: float, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._add_batch_history(Phase.VAL, epoch, batch_i, loss, y_pred, y_true)

    def add_test_batch(self, batch_i: int, y_pred: torch.Tensor, y_true: torch.Tensor):
        self._add_batch_history(Phase.TEST, 0, batch_i, 0, y_pred, y_true)
        self._test_predictions["pred"].append(self._ypred_to_binary(y_pred))
        self._test_predictions["true"].append(y_true)

    def add_prediction(self, img_ids: torch.tensor, y_pred: torch.tensor):
        """
        Adds a batch of predictions to the predictions dataframe.
        """
        combined = torch.hstack([img_ids.unsqueeze(1), self._ypred_to_binary(y_pred)])
        batch_prediction = pd.DataFrame(combined, columns=self._predictions.columns)
        self._predictions = pd.concat([self._predictions, batch_prediction], axis=0)

    def get_train_epoch_report(self, epoch: int) -> pd.Series:
        """
        Produce a single, aggregated statistics for a given epoch in the training phase.
        """
        return self._get_epoch_report(Phase.TRAIN, epoch)

    def get_validation_epoch_report(self, epoch) -> pd.Series:
        """
        Produce a single, aggregated statistics for a given epoch in the validation phase.
        """
        return self._get_epoch_report(Phase.VAL, epoch)

    def get_test_report(self) -> Dict[str, Any]:
        """
        Produces a dictionary with the aggregated statistics of the last epoch in the test phase.
        """
        return self._get_last_epoch_report(Phase.TEST)

    def get_train_history(self) -> pd.DataFrame:
        """
        Produces a Dataframe with aggregated statistics for all the available epochs in the training phase.
        """
        return self._get_history(Phase.TRAIN)

    def get_validation_history(self) -> pd.DataFrame:
        """
        Produces a Dataframe with aggregated statistics for all the available epochs in the validation phase.
        """
        return self._get_history(Phase.VAL)

    def get_test_history(self) -> pd.DataFrame:
        """
        Produces a Dataframe with aggregated statistics for the test phase (there is ony one "epoch").
        """
        return self._get_history(Phase.TEST)

    def get_predictions(self):
        """
        Returns the predictions dataframe. They don't have statistics, because we don't have access to the true labels.
        """
        return self._predictions.astype(int)

    def get_predictions_report(self):
        """
        Returns a confusion matrix for the test phase, and multilabel scores.
        """
        y_pred = torch.vstack(self._test_predictions["pred"])
        y_true = torch.vstack(self._test_predictions["true"])

        # Interpretation:
        #     - confusion_matrix[i, 0, 0]: true negatives for label i
        #     - confusion_matrix[i, 0, 1]: false positives for label i
        #     - confusion_matrix[i, 1, 0]: false negatives for label i
        #     - confusion_matrix[i, 1, 1]: true positive for label i
        confusion_matrix = skm.multilabel_confusion_matrix(y_true, y_pred)

        # Interpretation:
        #    - report[label, "metric"]: value of the metric for the given label
        report = skm.classification_report(y_true, y_pred, target_names=self.labels, output_dict=True, zero_division=0)
        return confusion_matrix, report

    def _get_last_epoch_report(self, phase: str) -> Dict[str, Any]:
        """
        Produces a dictionary with the report of the last epoch in the given phase.
        Useful for printing the results of each phase, which is what we're usually interested in.
        """
        # Need to compact the history before getting the report
        self._compact_history()

        if self._history[phase].empty:
            return {}

        last_epoch = self._history[phase]["epoch"].max()
        epoch_report = self._get_epoch_report(phase, last_epoch)

        # group all labels accuracies into a single dictionary:
        labels_report = epoch_report[self.labels].to_dict()
        report = epoch_report.drop(index=self.labels).to_dict()
        report["labels"] = labels_report

        return report

    def _get_epoch_report(self, phase: str, epoch: int) -> pd.Series:
        """
        Given an epoch and a phase, produces a single, aggregated statistics for the whole epoch.
        """
        # Need to compact the history before getting the epoch stats
        self._compact_history()

        # Get the whole recorded batch data for the given phase and epoch
        epoch_history = self._history[phase].groupby("epoch").get_group(epoch)

        # Accuracies need to be divided by the number of data points processed in the whole epoch
        accuracies = epoch_history[self.accuracy_columns].sum() / epoch_history["batch_size"].sum()
        accuracies = accuracies.rename(index={"correct": "accuracy"})

        # Other metrics are averaged over the batches
        avg_metrics = epoch_history[_METRICS].mean(axis=0)

        # Concatenate flattened statistics for the epoch
        return pd.concat([accuracies, avg_metrics])

    def _compact_history(self):
        """
        Concatenates the history of each batch into a single dataframe for each phase.
        Then, clean accumulated batch history
        """
        for phase, h in self._batch_history.items():
            if h:
                batch_aggregate = pd.DataFrame(h, columns=self._all_columns)
                self._history[phase] = pd.concat([self._history[phase], batch_aggregate], axis=0)
                self._batch_history[phase] = []

    def _add_batch_history(self, phase: str, epoch: int, batch_i: int, loss: float, y_pred: torch.Tensor,
                           y_true: torch.Tensor):
        """
        Calculate the metrics of a prediction and appends it to the history of the given phase.
        """
        self._batch_history[phase].append(
            [epoch, batch_i, y_pred.shape[0], loss, *self._calculate_batch_history_entry(y_pred, y_true)])

    def _ypred_to_binary(self, y_pred: torch.Tensor):
        # determines the number of correct samples and the number of times each label was correctly predicted
        return (y_pred > self._correct_threshold).float()

    def _calculate_batch_history_entry(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Given a batch of predictions and true labels, returns a list of metrics for this batch.
        """

        y_pred = self._ypred_to_binary(y_pred)

        correct_samples, correct_labels = self._count_corrects(y_pred, y_true)
        ma_p, ma_r, ma_f, _ = skm.precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        mi_p, mi_r, mi_f, _ = skm.precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        return [mi_p, mi_r, mi_f, ma_p, ma_r, ma_f, correct_samples] + correct_labels

    @staticmethod
    def _count_corrects(y_pred: torch.tensor, y_true: torch.tensor) -> (int, torch.tensor):
        """
        Given a batch of predictions and labels, returns the number of correct samples
        (samples having all labels correctly predicted) and the number times a label was correctly predicted
        (for each label).
        """
        correct_samples_in_batch = (y_pred == y_true).all(dim=1).sum().item()
        correct_labels_in_batch = (y_pred == y_true).float().sum(dim=0)

        return correct_samples_in_batch, correct_labels_in_batch.tolist()

    def _get_history(self, phase: str) -> pd.DataFrame:
        """
        Produces a Dataframe with aggregated statistics for all the available epochs in the given phase.
        """
        epochs = self._history[phase]["epoch"].unique()

        # Aggregates and concatenates epoch reports
        return pd.DataFrame([self._get_epoch_report(phase, epoch) for epoch in epochs])
