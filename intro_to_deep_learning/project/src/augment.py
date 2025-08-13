from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


class AugmentedImageDataset(Dataset):
    """
    Apply transformations to an image Dataset, to be used in training the model
    
    Arguments:
        dataset (Dataset): An image Dataset to be augmented that returns (sample, target, id)
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.resize = 128
        self.transforms = self.build_transforms()

    def __getitem__(self, idx):
        sample, target, img_id = self.dataset[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample, target, img_id

    def build_transforms(self):
        transformer = nn.Sequential(

            # resize
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25)), antialias=True),

            # image levels
            transforms.ColorJitter(hue=.02, saturation=.01),
            # transforms.ColorJitter(brightness=0.1),
            transforms.RandomRotation(10),

            # crop back to resize
            # transforms.CenterCrop(self.resize),
            transforms.RandomCrop(self.resize),

            # image flips
            # transforms.RandomVerticalFlip(0.4),
            transforms.RandomHorizontalFlip(0.4),
            
            # prevent overfitting
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.2, 2.2), value=0, inplace=False)
        )
        return transformer

    def __len__(self):
        return len(self.dataset)
