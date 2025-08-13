import os

import torch
import pandas as pd
from datetime import datetime

OUT_DIR = f"../results/{datetime.now():%m_%d_%H%M%S}/"


def ensure_path_exists(path: str):
    # Ensures that all the folders to the given path exists, and if not creates them
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return str(abs_path)


def training_step(model, optimizer, loss_function, loader, device):
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    len_t = len(loader)

    # Set the Model in training mode
    model.train()
    for i, batch in enumerate(loader):
        lengths = batch["lengths"]
        x = batch["inputs"]
        y = batch["labels"]
        x_data, y_target, lengths = x.to(device), y.to(device), lengths.to(device)

        y_pred = model(x_data, lengths)
        loss = loss_function(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_acc += (torch.argmax(y_pred, dim=1) == y_target).sum().item()
        epoch_loss += loss.item()
        total += len(x_data)

        if i % (len_t // 10) == 0:
            print(f"Progress {(i + 1) / len_t:.2%}: Loss: {epoch_loss / total}, Acc: {epoch_acc / total:.2%}, "
                  f"correct: {epoch_acc}/{total}")

    accuracy = epoch_acc / total
    loss = epoch_loss / (len(loader) + 1)
    return loss, accuracy


def validation_step(model, loss_function, loader, device):
    total_loss = 0
    total_correct = 0

    # set the model in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in loader:
            lengths = batch["lengths"]
            x = batch["inputs"]
            y = batch["labels"]
            x_data, y_target, lengths = x.to(device), y.to(device), lengths.to(device)

            # make the predictions and calculate the validation loss
            y_pred = model(x_data, lengths)
            total_loss += loss_function(y_pred, y_target).item()

            # calculate the number of correct predictions
            total_correct += (torch.argmax(y_pred, dim=1) == y_target).sum().item()

    accuracy = total_correct / len(loader.dataset)
    loss = total_loss / (len(loader) + 1)
    return loss, accuracy


def save_history(history):
    ensure_path_exists(OUT_DIR)
    file_history = OUT_DIR + "train_history.csv"
    df = pd.DataFrame(history, columns=["epoch", "train_accuracy", "train_loss", "val_accuracy", "val_loss"])
    df.to_csv(file_history, index=False)
    print(f"\nTrain history data saved as", file_history)


def save_test_result(test_acc, test_loss):
    ensure_path_exists(OUT_DIR)
    file_test_result = OUT_DIR + "test_result.csv"
    with open(file_test_result, "w") as file:
        file.write("accuracy,loss\n")
        file.write(",".join(map(str, [test_acc, test_loss])))
        file.write("\n")
    print("\nTest Results saved as", file_test_result)
