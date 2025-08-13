import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt
from typing import Dict, Any
from datetime import datetime
from torchvision import transforms, datasets

"""
Group: GradientDescendants
Group members:
Christian Cardin
Harri Nieminen
Elina Zetterman

In this exercise, we tried various combinations of models together with 
different choices for loss function and optimizer. In addition, we implemented
early stopping as a measure against overfitting. The script works by providing options via
command line:

-l [integer]: selects one of the available loss function by index. Possible values are:
    0: NLLLoss
    1: CrossEntropyLoss
    2: nn.MultiLabelMarginLoss
-o [integer]: selects one optimizer by index. Possible values are:
    0: SGD
    1: Adamax,
    2: Adam
-m [integer]: selects one of the available models by index. We only kept one model for the
              submission, to avoid unnecessary clutter. So this option has no effect

We tried multiple combinations, and the best of all was the current submitted model with 
Adamax optimizer and CrossEntropyLoss as loss function, and without the early stopping mechanism. The Resulting final accuracies are:
- Train set: 100.00%
- Validation: 91.37%
- Test: 91.20%

Which we deemed very good, and did not consider investigating further.
The other combinations were far inferior. Some were not better than random chance,
showing heavy underfitting. In general, most combinations did not converge at all after 
10 epochs, which we kept fixed for all tests. 

Below are results for some of the combinations.
With early stopping:
CrossEntropyLoss, Adamax:
Train acc 99.07%, validation acc 87.43%, test acc 88.86%, stop at epoch 5
CrossEntropyLoss, SGD: 
Train acc 4.32%, validation acc 3.74%, test acc 3.19%, stop at epoch 1
CrossEntropyLoss, Adam: 
Train acc 95.26%, validation acc 79.97%, test acc 79.90%, stop at epoch 2
NLLLoss, Adamax:
Train acc 4.52%, validation acc 3.00%, test acc 2.83%, no early stop

Without early stopping:
CrossEntropyLoss, Adamax:
Train acc 100.00%, validation acc 91.37%, test acc 91.20%
NLLLoss, SGD:
Train acc 4.10%, validation acc 4.40%, test acc 4.82%
CrossEntropyLoss, Adam:
Train acc 97.53%, validation acc 78.06%, test acc 77.81%
"""

# --- hyperparameters ---
N_EPOCHS = 10
BATCH_SIZE_TRAIN = 100
BATCH_SIZE_TEST = 100
LR = 0.01

# --- fixed constants ---
NUM_CLASSES = 24
DATA_DIR = '../data/sign_mnist_%s'
OUT_DIR = f"../results/{datetime.now():%m_%d_%H%M%S}/"


def ensure_path_exists(path: str):
    # Ensures that all the folders to the given path exists, and if not creates them
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return str(abs_path)


def read_cmd_args() -> Dict[str, Any]:
    args = {}
    for i in range(1, len(sys.argv), 2):
        # check if the argument has a corresponding value
        if i + 1 < len(sys.argv):
            # add the argument and its value to the dictionary
            args[sys.argv[i]] = int(sys.argv[i + 1])
    return args


def get_data_loaders():
    # We transform image files' contents to tensors
    # Plus, we can add random transformations to the training data if we like
    # Think on what kind of transformations may be meaningful for this data.
    # Eg., horizontal-flip is definitely a bad idea for sign language data.
    # You can use another transformation here if you find a better one.
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor()])

    train_set = datasets.ImageFolder(DATA_DIR % 'train', transform=train_transform)
    dev_set = datasets.ImageFolder(DATA_DIR % 'dev', transform=test_transform)
    test_set = datasets.ImageFolder(DATA_DIR % 'test', transform=test_transform)

    # Create Pytorch data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE_TEST, shuffle=False)

    return train_loader, dev_loader, test_loader


class CnnModel(nn.Module):
    def __init__(self, num_classes: int, size: int = 28, model_selection: int = 0):
        super(CnnModel, self).__init__()
        self.out_features = num_classes
        self.picture_size = size
        self.layers = self.model_1()

    def forward(self, x):
        return self.layers(x)

    def model_1(self):
        return nn.Sequential(
            # 1sr Convolutional Layer
            # in: (batch_size, 1, size, size)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            # out: (batch_size, 16, size, size)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out: (batch_size, 16, size/2, size/2)

            # 2nd Convolutional Layer
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            # out: (batch_size, 32, size/2, size/2)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out: (batch_size, 32, size/4, size/4)

            # 3rd Convolutional Layer
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # out: (batch_size, 64, size/4, size/4)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out: (batch_size, 64, size/8, size/8)

            # 4th Convolutional Layer
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            # out: (batch_size, 128, size/8, size/8)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # out: (batch_size, 128, size/16, size/16)

            # Final Layer (start_dim is necessary to preserve minibatch dimension)
            nn.Flatten(start_dim=1),
            # out: (batch_size, 128 * size/16 * size/16)
            nn.Linear(in_features=128 * (self.picture_size // 16) * (self.picture_size // 16), out_features=self.out_features),
            # out: (batch_size, out_features)
            nn.ReLU(),
            nn.Linear(in_features=self.out_features, out_features=self.out_features)
            # out: (batch_size, out_features)
        )


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    # the class method early_stop returns True if the error has grown more than min_delta
    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_device():
    # --- set up ---
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_model(picture_size: int, cmd_args: Dict, device):
    return CnnModel(
        num_classes=NUM_CLASSES,
        size=picture_size,
        model_selection=cmd_args.get("-m", 1)).to(device)


def get_optimizer(model, cmd_args: Dict):
    optimizers = [optim.SGD(model.parameters(), lr=LR),
                  optim.Adamax(model.parameters(), lr=LR),
                  optim.Adam(model.parameters(), lr=LR)]
    return optimizers[cmd_args.get("-o", 1)]


def get_loss_function(cmd_args: Dict):
    loss_functions = [nn.NLLLoss(), nn.CrossEntropyLoss(), nn.MultiLabelMarginLoss()]
    return loss_functions[cmd_args.get("-l", 1)]


def get_picture_shape(loader):
    # [color_channels, height, width]
    return loader.dataset[0][0].shape


def show_data(loader):
    # look at loaded data, first batch; train set
    images, labels = next(iter(loader))
    img = torchvision.utils.make_grid(images)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.show()


def training_step(epoch, train_loader, model, loss_function, optimizer, device):
    train_loss = 0
    train_correct = 0
    total = 0

    # Set the Model in training mode
    model.train()

    for batch_num, (x_data, y_target) in enumerate(train_loader):
        x_data, y_target = x_data.to(device), y_target.to(device)

        y_pred = model(x_data)
        loss = loss_function(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_correct += (torch.argmax(y_pred, dim=1) == y_target).sum().item()
        train_loss += loss.item()
        total += len(x_data)

        print('Training: Epoch %d - Batch %d/%d: Loss: %.4f | Train Acc: %.3f%% (%d/%d)' %
              (epoch, batch_num, len(train_loader), train_loss / (batch_num + 1),
               100. * train_correct / total, train_correct, total))

    accuracy = train_correct / total
    loss = train_loss / (len(train_loader) + 1)
    return accuracy, loss


def validation_step(dev_loader, model, loss_function, device):
    total_loss = 0
    total_correct = 0

    # set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        for (x_data, y_target) in dev_loader:
            x_data, y_target = x_data.to(device), y_target.to(device)

            # make the predictions and calculate the validation loss
            y_pred = model(x_data)
            total_loss += loss_function(y_pred, y_target).item()

            # calculate the number of correct predictions
            total_correct += (torch.argmax(y_pred, dim=1) == y_target).sum().item()

    accuracy = total_correct / len(dev_loader.dataset)
    loss = total_loss / (len(dev_loader) + 1)
    return accuracy, loss


def test_step(test_loader, model, loss_function, device):
    total_loss = 0
    total_correct = 0
    total = 0

    # set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch_num, (x_data, y_target) in enumerate(test_loader):
            x_data, y_target = x_data.to(device), y_target.to(device)
            y_pred = model(x_data)

            total_loss += loss_function(y_pred, y_target).item()
            total_correct += (torch.argmax(y_pred, dim=1) == y_target).sum().item()
            total += len(x_data)

            print('Evaluating: Batch %d/%d: Loss: %.4f | Test Acc: %.3f%% (%d/%d)' %
                  (batch_num, len(test_loader), total_loss / (batch_num + 1),
                   100. * total_correct / total, total_correct, total))

    accuracy = total_correct / total
    loss = total_loss / (len(test_loader) + 1)
    return accuracy, loss


def save_history(history):
    file_history = OUT_DIR + "train_history.csv"
    df = pd.DataFrame(history, columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
    df.to_csv(file_history, index=False)
    print(f"\nTrain history data saved as", file_history)


def save_test_result(test_acc, test_loss):
    file_test_result = OUT_DIR + "test_result.csv"
    with open(file_test_result, "w") as file:
        file.write("accuracy,loss\n")
        file.write(",".join(map(str, [test_acc, test_loss])))
        file.write("\n")
    print("\nTest Results saved as", file_test_result)


def save_model(model):
    model_name = OUT_DIR + "model.pt"
    torch.save(model, model_name)
    print("\nModel Saved as", model_name)


def main():
    abs_out = ensure_path_exists(OUT_DIR)
    print("Absolute path for output files:", str(abs_out))
    device = get_device()
    train_loader, dev_loader, test_loader = get_data_loaders()
    pic_shape = get_picture_shape(train_loader)
    picture_size = pic_shape[1]  # assumes squared pictures
    cmd_params = read_cmd_args()

    # show_data()
    model = get_model(picture_size, cmd_params, device)
    loss_function = get_loss_function(cmd_params)
    optimizer = get_optimizer(model, cmd_params)
    early_stopper = EarlyStopper()

    history = []
    print("** TRAINING STEP **")
    for epoch in range(N_EPOCHS):
        train_accuracy, train_loss = training_step(epoch, train_loader, model, loss_function, optimizer, device)

        print("Calculate Accuracy on Validation set...")
        val_accuracy, val_loss = validation_step(dev_loader, model, loss_function, device)

        # Collect and print data
        print(f"-- END EPOCH {epoch} --")
        print(f"\t* Train Accuracy: {train_accuracy:.2%}, Loss: {train_loss:.4f}")
        print(f"\t* Validation Accuracy: {val_accuracy:.2%}, Loss: {val_loss:.4f}")
        history.append([epoch, train_accuracy, train_loss, val_accuracy, val_loss])

        # Check for early stop condition
        if early_stopper.early_stop(val_loss):
            print("!! EARLY STOP !!")
            break

    print(f"-- END TRAINING --")
    save_history(history)
    save_model(model)

    # Final Test
    print("** TESTING STEP **")
    test_accuracy, test_loss = test_step(test_loader, model, loss_function, device)
    print(f"Final Accuracy: {test_accuracy:.2%}, Loss: {test_loss:.4f}")
    save_test_result(test_accuracy, test_loss)


if __name__ == "__main__":
    main()
