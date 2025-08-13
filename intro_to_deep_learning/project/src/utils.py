import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn

import common
import data
import getters
from metrics_collector import MetricsCollector

OUT_DIR = f"../results/"
OUT_DIR_PREDICTIONS = f"../predictions/"
CMD_ARGS_FILENAME = "args.json"


# namedtuple for storing the test result:
def ensure_path_exists(path: str):
    # Ensures that all the folders to the given path exists, and if not creates them
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return str(abs_path)


def show_data(loader, labels_map, category_to_visualize: str):
    """
    Shows a grid of images from the given loader, belonging to the given category.
    """
    img_grid = []
    grid_size = 5
    total_images = grid_size ** 2

    # Get 25 images from the loader, belonging to the given category
    for images, labels, img_id in loader:
        img_grid.extend([(im, lab, im_id) for im, lab, im_id in zip(images, labels, img_id)
                         if torch.any(lab * labels_map[category_to_visualize])])
        if len(img_grid) >= total_images:
            break

    # Print the labels of the images
    print(f"* {category_to_visualize}: {labels_map[category_to_visualize]}")
    for i, (im, lab, img_id) in enumerate(img_grid):
        print(f"{i} - {img_id}: {lab}")

    # Show the images
    img = torchvision.utils.make_grid([i for i, _, __ in img_grid[:total_images]], nrow=grid_size)
    plt.tight_layout()
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def print_cmd_args(model: nn.Module, loss_function: nn.Module, optimizer: torch.optim.Optimizer,
                   cmd_args: Dict[str, Any]):
    print("Command line arguments: ", cmd_args)
    print("ID:", cmd_args.get("-id") or "--")
    print("Model: ", model.__class__.__name__)
    print("Loss function: ", loss_function.__class__.__name__)
    print("Optimizer: ", optimizer.__class__.__name__)
    print("Color:", "RGB" if cmd_args.get("-rgb") else "Grayscale")
    print("Exclude unlabeled images:", cmd_args.get("-noe"))
    print("Batch size:", cmd_args.get("-b"))
    print("Dataset Size:", cmd_args.get("-s"))
    print("Data split for training:", cmd_args.get("-split"))
    print("Use separate test dataset for test step:", cmd_args.get("-test"))
    print("Correctness Threshold:", cmd_args.get("-t"))
    print("Learning rate:", cmd_args.get("-lr"))
    print("Epochs:", cmd_args.get("-e"))
    print("Verbose:", cmd_args.get("-v"))
    print("Cache images:", cmd_args.get("-cache"))


def _save_args(cmd_args, out_path):
    file_args = out_path + CMD_ARGS_FILENAME
    with open(file_args, "w") as f:
        json.dump(cmd_args, f, indent=4)


def _save_model(model, out_path):
    if model is None:
        return
    file_model = out_path + "model.pt"
    torch.save(model.state_dict(), file_model)
    print("Model saved as", file_model)


def load_model_from_disk(path: str, device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a neural network model from a pre-trained model file.
    Also, read and return the command line arguments used to train the model.
    """
    with open(path + CMD_ARGS_FILENAME, "r") as f:
        cmd_args = json.load(f)

    model_weights = torch.load(path + "model.pt", map_location=device)
    model = getters.get_model(cmd_args, device)
    model.load_state_dict(model_weights)
    return model, cmd_args


def save_prediction_data(metrics: MetricsCollector, cmd_args: Dict[str, Any]):
    """
    Extract the predictions from the metrics collector and save them to the output.
    Creates the following files:
    - predictions.tsv: a tsv file containing the predictions of the model on the prediction dataset
    - args.json: the command line arguments used to run the experiment
    """
    folder_name = _get_folder_name(cmd_args, test_accuracy=0)

    out_path = OUT_DIR + folder_name
    ensure_path_exists(out_path)

    _generate_prediction_file(metrics.get_predictions(), out_path)
    _save_args(cmd_args, out_path)
    print("Output dir:", out_path)


def save_training_and_evaluation_data(metrics: MetricsCollector, model: Optional[nn.Module], cmd_args: Dict[str, Any]):
    """
    Saves all the data from the given metrics collector to the results folder.
    It handles both training data and evaluation data contained in the metrics collector. It produces the following files:
    - args.json: the command line arguments used to run the experiment
    - model.pt: the model weights
    - train_history.png: a plot of the training history, including the loss, accuracy and other metrics
    - test_dataset_report.json: a json file containing a summary of metrics for each label (precision, recall, f1)
    - confusion_matrix_test.png: a plot of the confusion matrix
    - predictions.tsv: a tsv file containing the predictions of the model on the prediction dataset
    """
    prefix = cmd_args.get("-id") or ""
    mode = cmd_args.get("-mode")
    test_report = metrics.get_test_report()
    confusion_matrix, prediction_report = metrics.get_predictions_report()

    # Produce an output folder with summary info of the hyperparameters
    m = cmd_args.get("-m")
    lr = str(cmd_args.get("-lr")).split(".")[-1]
    e = cmd_args.get("-e")
    test_accuracy = test_report["accuracy"]
    folder_name = f"{prefix}_{mode}_{test_accuracy * 100:.0f}_m{m}_lr{lr}_e{e}_{datetime.now():%m%d%H%M%S}/"

    out_path = OUT_DIR + folder_name
    ensure_path_exists(out_path)

    if metrics.has_train_history:
        train_history = metrics.get_train_history()
        val_history = metrics.get_validation_history()
        _plot_history(train_history, val_history, out_path)
        _save_model(model, out_path)

    _generate_prediction_file(metrics.get_predictions(), out_path)
    _save_test_report(prediction_report, out_path)
    _generate_confusion_plot(confusion_matrix, common.LABELS, out_path)
    _save_args(cmd_args, out_path)
    print("Output dir:", out_path)


def _get_folder_name(cmd_args: Dict[str, Any], test_accuracy: float) -> str:
    """
     Produce an output folder name, giving a meaningful summary info of the hyperparameters
    """
    prefix = cmd_args.get("-id") or ""
    mode = cmd_args.get("-mode")
    m = cmd_args.get("-m")
    lr = str(cmd_args.get("-lr")).split(".")[-1]
    e = cmd_args.get("-e")
    return f"{prefix}_{mode}_{test_accuracy * 100:.0f}_m{m}_lr{lr}_e{e}_{datetime.now():%m%d%H%M%S}/"


def _plot_history(train_history: pd.DataFrame, val_history: pd.DataFrame, out_path: str):
    """
    Plots the train and validation history to a file
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))

    # Plot train and validation Accuracy on the same axis:
    axes[0, 0].plot(train_history["accuracy"], color="blue", label="Train Accuracy")
    axes[0, 0].plot(val_history["accuracy"], color="orange", label="Validation Accuracy")
    axes[0, 0].plot(val_history["micro_precision"], color="green", label="Micro Precision")
    axes[0, 0].plot(val_history["macro_precision"], color="purple", label="Macro Precision")
    axes[0, 0].set_title("Train and Validation Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].legend(loc="best")

    # Plot train and validation Loss on the same axis:
    axes[0, 1].plot(train_history["loss"], color="blue", label="Train Loss")
    axes[0, 1].plot(val_history["loss"], color="orange", label="Validation Loss")
    axes[0, 1].set_title("Train and Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(loc="best")

    # Plot each label accuracy on the same axis:
    for label in common.LABELS:
        axes[1, 0].plot(train_history[label], label=label)
        axes[1, 1].plot(val_history[label], label=label)

    axes[1, 0].set_title("Labels Train Accuracy")
    axes[1, 1].set_title("Labels Validation Accuracy")
    for ax in axes[1, :]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend(loc="best")

    # Plot precision, recall and f1-score
    for key in ["micro_precision", "micro_recall", "micro_f1", "macro_precision", "macro_recall", "macro_f1"]:
        axes[2, 0].plot(train_history[key], label=key)
        axes[2, 1].plot(val_history[key], label=key)

    axes[2, 0].set_title("Train Metrics")
    axes[2, 1].set_title("Validation Metrics")
    for ax in axes[2, :]:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend(loc="best")

    file_plot = out_path + "train_history.png"
    plt.tight_layout()
    plt.savefig(file_plot)
    print("\nPlot saved as", file_plot)


def _generate_confusion_plot(conf_matrix: np.ndarray, labels: np.ndarray, out_path: str):
    """
    Plots the confusion matrix to a file
    """
    file_plot = out_path + "confusion_matrix_test.png"
    fig, ax = plt.subplots(7, 2, figsize=(9, 17))

    for axes, cfs_matrix, label in zip(ax.flatten(), conf_matrix, labels):
        _confusion_matrix_subplot(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.savefig(file_plot)
    print("\nConfusion plot saved as", file_plot)


def _confusion_matrix_subplot(confusion_matrix, axes, class_label, class_names, fontsize=14):
    """
    Builds the confusion matrix frame
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("Confusion Matrix for the class - " + class_label)


def _generate_prediction_file(predictions: pd.DataFrame, out_path: str):
    """
    Generates a file with the predictions of the test set, as specified in the assignment.
    """
    file_predictions = out_path + "predictions.tsv"
    # sort the predictions by the image id
    predictions = predictions.sort_values(by="Filename")
    predictions.to_csv(file_predictions, index=False, sep="\t")
    print("\nPredictions saved as", file_predictions)


def _save_test_report(prediction_report, out_path: str):
    """
    Saves the final test results to a file
    """
    file_prediction_report = out_path + "test_dataset_report.json"
    with open(file_prediction_report, "w") as file:
        json.dump(prediction_report, file, indent=4)

    print("\nTest Results saved as", file_prediction_report)


def print_labels_distribution(annotations_path: str, images_path: str):
    labels_map = data.read_annotations(annotations_path, images_path, exclude_unlabelled=False)
    all_labels = torch.vstack(list(labels_map.values()))

    labels_frequency = all_labels.sum(dim=0).tolist()
    labels_count_per_image = all_labels.sum(dim=1).tolist()
    unlabeled_count = labels_count_per_image.count(0)
    labelled_count = len(labels_count_per_image) - unlabeled_count

    print("Total images:", len(labels_map))
    print("Unlabeled images:", unlabeled_count)
    print("Labeled images:", labelled_count)
    print("Ratio", labelled_count / len(labels_map))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.bar(common.LABELS, labels_frequency)
    ax1.set_title("Labels Frequency")
    ax1.set_xlabel("Label")
    ax1.set_ylabel("Frequency")

    ax2.hist(labels_count_per_image)
    ax2.set_title("Labels Count Per Image")
    ax2.set_xlabel("Labels Count")
    ax2.set_ylabel("Images Count")

    plt.tight_layout()
    plt.show()
