import os
import re
from typing import Dict, Tuple, List

import numpy as np

from augment import AugmentedImageDataset

import torch
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torchvision.io import ImageReadMode, read_image

from common import LABELS, LABELS_ENCODED


class ImageDataset(Dataset):
    """
    Custom dataset for the images.
    IMPORTANT: The images are cached in memory for speeding up the training,
    so make sure you have enough RAM. There's the option to disable this behaviour with
    the cmd argument "-cache 0"
    """

    def __init__(self, data_folder, annotations, cmd_args):
        self.data_folder = data_folder
        # NOTE: IDs go from 1 to N, we need to take this into account when indexing from cache
        self.ids = list(annotations.keys())
        self.labels = [v.float() for v in annotations.values()]
        self.cache = [None] * len(annotations)
        self.transformations = _get_transformations(cmd_args)
        self.use_cache = cmd_args.get("-cache")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item_index):
        # NOTE: img_id actually is item_index+1 in this project, but we don't want to give it for granted.
        img_id = self.ids[item_index]
        label = self.labels[item_index]

        # Cache image value if not already cached
        if self.use_cache and self.cache[item_index] is not None:
            img = self.cache[item_index]
        else:
            img = read_image(f"{self.data_folder}/im{img_id}.jpg", ImageReadMode.RGB)

            # rescale RGB values to interval [0-1] and apply transformations
            img = img.float() / 255
            img = self.transformations(img)

            # assign to cache if enabled
            if self.use_cache:
                self.cache[item_index] = img

        # return image cached data, its label and id
        return img, label, img_id


class ImageDatasetNoLabels(Dataset):
    """
    Custom dataset for loading the images, no labels provided
    """

    def __init__(self, data_folder, cmd_args):
        self.data_folder = data_folder
        self.img_ids = np.sort(_get_images_ids(data_folder))
        self.transformations = _get_transformations(cmd_args)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item_index):
        img_id = self.img_ids[item_index]
        img = read_image(f"{self.data_folder}/im{img_id}.jpg", ImageReadMode.RGB)

        # rescale RGB values to interval [0-1] and apply transformations
        img = self.transformations(img.float() / 255)
        return img, img_id


def _get_transformations(cmd_args: Dict) -> nn.Sequential:
    """
    Returns the transformations to be applied to the images.
    :param cmd_args: command line arguments
    :return: the transformations to be applied to the images
    """
    transformations = nn.Sequential(
        # This normalization is based on the ImageNet, and it's a common practice for image classification
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    )
    if not cmd_args.get("-rgb"):
        transformations.append(transforms.Grayscale())
    return transformations


def get_dataset_no_labels(data_folder, cmd_args) -> Tuple[DataLoader, Dict[str, torch.tensor]]:
    """
    Loads the image data from the given folder, without labels.
    Returns the Dataloader and a dictionary mapping text labels to their one-hot encoded vectors
    """
    dataset = ImageDatasetNoLabels(data_folder, cmd_args)
    return DataLoader(dataset, batch_size=cmd_args.get("-b")), LABELS_ENCODED


def get_full_dataset_with_labels(annotations_path: str, image_path: str, cmd_args: Dict) -> Tuple[DataLoader, Dict[str, torch.tensor]]:
    """
    Reads the data files from the given paths.
    Associates each image with its labels from the annotation data, in form of one-hot encoded vectors.
    If the cmd argument "-noe 1" is passed, images without labels are excluded from the dataset.

    :param annotations_path: path to the annotations file
    :param image_path: path to the folder containing the images
    :param cmd_args: command line arguments
    returns: the full dataset and a dictionary mapping text labels to their one-hot encoded vectors
    """
    batch_size = cmd_args.get("-b")
    exclude_unlabeled = cmd_args.get("-noe")

    annotations = read_annotations(annotations_path, image_path, exclude_unlabeled)
    dataset = ImageDataset(image_path, annotations, cmd_args)

    # Limit dataset to a subset of data
    dataset_size = cmd_args.get("-s")
    if dataset_size < 1:
        dataset, _ = random_split(dataset, [dataset_size, 1 - dataset_size])

    return DataLoader(dataset, batch_size=batch_size), LABELS_ENCODED


def get_splitted_dataset(annotations_path: str, image_path: str, cmd_args: Dict) -> Tuple[
    DataLoader, DataLoader, DataLoader, Dict[str, torch.tensor]]:
    """
    Loads the dataset with annotation, and prepares it for training splitting it in train, dev and test sets.
    Additionally, wraps the train dataset in an augmented dataset that performs random transformations on the images.
    returns: train, dev, test sets, and a dictionary mapping text labels to their one-hot encoded vectors
    """
    batch_size = cmd_args.get("-b")
    train_split = cmd_args.get("-split")
    use_test_dataset = cmd_args.get("-test")
    augment_dataset = cmd_args.get("-aug")

    test_dev_split = (1-train_split)/2
    dataloader, text_labels_map = get_full_dataset_with_labels(annotations_path, image_path, cmd_args)

    # Split into train, dev, test sets
    if use_test_dataset:
        train, dev, test = random_split(dataloader.dataset, lengths=[train_split, test_dev_split, test_dev_split])
    else:
        # Split using train and dev sets only, use dev for testing
        train, dev = random_split(dataloader.dataset, lengths=[train_split, 1-train_split])
        test = dev

    # apply data augmentation to training dataset
    if augment_dataset:
        train = AugmentedImageDataset(train)

    tr = DataLoader(train, batch_size=batch_size, shuffle=True)
    de = DataLoader(dev, batch_size=batch_size)
    te = DataLoader(test, batch_size=batch_size)
    return tr, de, te, text_labels_map


def _count_files_by_extension(path: str, extension: str) -> int:
    """
    Counts the number of files in the given path that have the given extension.
    """
    return len([filename for filename in os.listdir(path) if filename.endswith(extension)])


def _get_images_ids(images_path: str) -> List[int]:
    """
    Given the path to the images folder, read all the image ids contained in the files.
    """
    id_regex = re.compile(r"im(\d+).jpg")
    return [int(id_regex.match(filename).group(1)) for filename in os.listdir(images_path) if
            filename.endswith(".jpg")]


def read_annotations(annotations_path: str, images_path: str, exclude_unlabelled: bool) -> Dict[int, torch.tensor]:
    """
    Given the path to the annotation folder, read all the image ids contained in the annotation files.
    Produces a dictionary where the keys are the image IDs, and the values are the categorical one-hot encoded tensor
    for that image. For example, if the one-hot-encoded array for "tree" = [0, 0, 0, 1] and "dog" = [0, 0, 1, 0] and
    the picture "666.png" depicts a dog under a tree, its entry in the dictionary will be:

    data[666] = torch.tensor[0, 0, 1, 1]
    """

    # All annotation file paths
    annotation_files = [os.path.join(annotations_path, filename)
                        for filename in os.listdir(annotations_path)
                        if filename.endswith(".txt")]

    number_of_categories = len(LABELS)
    image_ids = _get_images_ids(images_path)

    # initialize data with ids as keys, and empty tensors as values
    if exclude_unlabelled:
        data = {}
    else:
        data = {img_id: torch.zeros(number_of_categories, dtype=torch.uint8) for img_id in image_ids}

    # image_id --> categorical one-hot encoded tensor
    for i, file_path in enumerate(annotation_files):
        label = os.path.basename(file_path).split(".")[0]
        with open(file_path, "r") as f:
            for image_id in map(int, f.readlines()):
                if image_id not in data:
                    data[image_id] = torch.zeros(number_of_categories)

                data[image_id] += LABELS_ENCODED[label]

    return data
