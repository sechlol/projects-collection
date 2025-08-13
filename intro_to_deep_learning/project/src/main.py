import click
import torch
import data
import getters
import utils

from gym import Gym
from typing import Dict
from metrics_collector import MetricsCollector

# Constants
IMAGE_PATH = "../data/images"
IMAGE_EVALUATION_PATH = "../data/eval_images"
ANNOTATIONS_PATH = "../data/annotations"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_cmd_args(overrides: Dict = None):
    """
    Read command line arguments with click.
    """
    args = {}

    @click.command()
    @click.option("-m", default=4, type=click.IntRange(0, 6), help="Model version")
    @click.option("-loss", default=0, type=click.IntRange(0, 3), help="Loss function type")
    @click.option("-opt", default=4, type=click.IntRange(0, 4), help="Optimizer type")
    @click.option("-rgb", default=True, help="If True, load image in RGB. Else, in Grayscale")
    @click.option("-noe", default=False, help="If true, exclude images with empty labels from training")
    @click.option("-v", default=False, help="Enable verbose output")
    @click.option("-b", default=32, help="Batch size")
    @click.option("-lr", default=0.2, help="Learning rate")
    @click.option("-e", default=3, help="Number of epochs")
    @click.option("-t", default=0.5, help="Threshold for prediction")
    @click.option("-s", default=0.1, type=click.FloatRange(0, 1, min_open=True), help="Fraction of dataset to load.")
    @click.option("-aug", default=False, help="If true, enable data augmentation for training images")
    @click.option("-split", default=0.85, type=click.FloatRange(0, 1, min_open=True, max_open=True), help="Fraction of data to use for training.")
    @click.option("-cache", default=True, help="If true, cache images in memory for faster training")
    @click.option("-test", default=False, help="If False, use the the dev dataset also for testing.")
    @click.option("-mode", default="train", type=click.Choice(["train", "eval", "pred"], case_sensitive=False), help="Operating mode")
    @click.option("-load", default=None, type=click.Path(exists=True), help="Load model for evaluation or prediction. It needs to point to a folder containing a model.pt file")
    @click.option("-id", default=None, help="If given, adds an id to the final folder name")
    def read_args(m, loss, opt, rgb, noe, v, b, lr, e, t, s, aug, split, cache, test, mode, load, id):
        args.update(
            {"-m": m, "-opt": opt, "-loss": loss, "-rgb": rgb, "-b": b, "-lr": lr, "-e": e, "-s": s, "-v": v,
             "-t": t, "-load": load, "-noe": noe, "-split": split, "-cache": cache, "-test": test, "-id": id,
             "-aug": aug, "-mode": mode})

    overrides = overrides or {}
    try:
        read_args()
    except SystemExit as e:
        # Unless called from main(), the click.command() will raise SystemExit.
        # Therefore, we just catch it and continue normally
        pass

    args.update(overrides)
    return args


def train_and_evaluate(model, cmd_args):
    """
    Train and test a model from scratch. Saves the results in a folder
    """
    train_loader, dev_loader, test_loader, labels_map = data.get_splitted_dataset(ANNOTATIONS_PATH, IMAGE_PATH,
                                                                                  cmd_args)
    loss_function = getters.get_loss_function(cmd_args)
    optimizer = getters.get_optimizer(model, cmd_args)

    # Print information. Uncomment some of these lines if you want to see additional plots
    utils.print_cmd_args(model, loss_function, optimizer, cmd_args)
    # utils.show_data(train_loader, labels_map, "female")
    # utils.print_labels_distribution(ANNOTATIONS_PATH, IMAGE_PATH)

    # Initialize gym
    metrics = MetricsCollector(labels=list(labels_map.keys()), cmd_args=cmd_args)
    gym = Gym(model, loss_function, optimizer, metrics, cmd_args, DEVICE)

    # Train & Test model
    print("\n\n** BEGIN TRAINING **")
    gym.train_model(train_loader, dev_loader)

    print("\n\n** BEGIN TESTING **")
    gym.test_model(test_loader)

    utils.save_training_and_evaluation_data(metrics, model, cmd_args)


def evaluate(model, cmd_args):
    """
    Make predictions on a single labeled dataset. Compare the predictions to the ground truth and save the results
    """

    data_loader, labels_map = data.get_full_dataset_with_labels(ANNOTATIONS_PATH, IMAGE_PATH, cmd_args)
    metrics = MetricsCollector(labels=list(labels_map.keys()), cmd_args=cmd_args)

    # Initialize gym. Because it only needs to run evaluation, it doesn't need the loss or the optimizer
    gym = Gym(model, loss_function=None, optimizer=None, metrics=metrics, cmd_args=cmd_args, device=DEVICE)

    # Evaluate model on the whole dataset
    print("\n\n** BEGIN EVALUATION **")
    gym.test_model(test_loader=data_loader)

    # save results
    utils.save_training_and_evaluation_data(metrics, model, cmd_args)


def predict(model, cmd_args):
    data_loader, labels_map = data.get_dataset_no_labels(IMAGE_EVALUATION_PATH, cmd_args)
    metrics = MetricsCollector(labels=list(labels_map.keys()), cmd_args=cmd_args)

    # Initialize gym. Because it only needs to run evaluation, it doesn't need the loss or the optimizer
    gym = Gym(model, loss_function=None, optimizer=None, metrics=metrics, cmd_args=cmd_args, device=DEVICE)

    # Make prediction on the evaluation dataset
    print("\n\n** BEGIN PREDICTION **")
    gym.predict(data_loader=data_loader)

    # save results
    utils.save_prediction_data(metrics, cmd_args)


def main():
    # Fetch basic dependencies
    cmd_args = read_cmd_args()
    mode = cmd_args.get("-mode")
    model_path = cmd_args.get("-load")

    if mode == "train":
        model = getters.get_model(cmd_args, device=DEVICE)
        train_and_evaluate(model, cmd_args)
    else:
        assert model_path, "You must provide a model path to run in eval or pred mode. Use -load <path> option to do so."

        # load pre-trained model from disk and its corresponding cmd_args
        model, model_cmd_args = utils.load_model_from_disk(model_path, device=DEVICE)

        # For development, we can override some of the cmd_args (like the dataset size, for faster debug)
        for arg in ["-s", "-mode", "-v"]:
            model_cmd_args[arg] = cmd_args.get(arg)

        if mode == "eval":
            evaluate(model, model_cmd_args)
        elif mode == "pred":
            predict(model, model_cmd_args)
        else:
            raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    main()
