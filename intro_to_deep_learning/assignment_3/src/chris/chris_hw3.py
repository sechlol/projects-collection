# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings

   Hande Celikkanat

   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim

import data
import utils

"""
Group: GradientDescendants
Members: 
    * Christian Cardin
    * Harri Nieminen
    * Elina Zetterman
    
In this assignment we have modified the base implementation to add extra functionalities, to make the 
development and hyperparameter tuning easier and faster. In particular, we've done two mayor changes:

1. Hyperparameters are parametrized via command line arguments
    Most hyperparameters can be provided via command line. In this way it's possible to schedule multiple
    jobs in Puhti at the same time, with different parameters. The parameters change things like model's type
    (rnn/lstm/gru), hidden size, number of recurrent layers and nonlinearity function. In addition, we can also control
    the number of epochs, learning rate, loss function, the optimizer and the batch size.
     
2. TwitterDataset class was optimized for computational performance
    We've noticed that the TwitterDataset.__getitem__() function was running the tokenization and token encoding
    each time, when this operation could be run only once and reuse its results. TwitterDataset now compute the
    tweet's encodings in the constructor, and also computes the lengths array. This will result in a longer loading
    time, but the performance boost for the training step is massive.
    In addition, we added a functionality for custom sorting the dataset to reduce the need of padding as suggested
    in the assignment instructions: the data is shuffled and then sorted by lengths in descending order. In this way
    the batch will contain sequence of mostly the same length, reducing the need for padding and speeding up 
    the packing step. To save a bit of computational time, the shuffling operation is not performed on the data arrays 
    directly; instead, we shuffle an index vector that will be used to reference the elements in the original arrays.
    

In our experiments we found that LSTM is the type of RNN that gives better accuracy, which hidden size around 15.
Also, adding a Pooling layer after the RNN seem to provide a considerable boost in accuracy. 

We were able to reach the following accuracies on the test set:
    * 70% on small dataset, 
    * 75% on medium dataset 
    * 78% on large dataset
"""

# Constants defaults
LR = 0.01
N_EPOCHS = 5
BATCH_SIZE = 32
EMBEDDING_DIM = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEV_FILE = '../data/sent140.dev.csv'
TEST_FILE = '../data/sent140.test.csv'
TRAIN_FILE = {
    0: '../data/sent140.train.mini.csv',
    1: '../data/sent140.train.midi.csv',
    2: '../data/sent140.train.csv'
}


# Recurrent network definition
class RNN1(nn.Module):
    def __init__(self, embedding_matrix, cmd_args):
        super(RNN1, self).__init__()
        input_size = embedding_matrix.shape[1]
        hidden_size = cmd_args.get("-h", 15)
        n_layers = cmd_args.get("-rnn_l", 1)
        rnn_layer = cmd_args.get("-rnn_t", "rnn")
        use_lstm = rnn_layer == "lstm"
        use_gru = rnn_layer == "gru"
        use_tanh = "-rnn_tanh" in cmd_args

        if use_lstm:
            self.recurrent_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        elif use_gru:
            self.recurrent_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers)
        else:
            self.recurrent_layer = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                nonlinearity="tanh" if use_tanh else "relu")

        self.embed = nn.Embedding.from_pretrained(embedding_matrix)
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.ffnn = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2),
            nn.ReLU(),
            nn.LogSoftmax(dim=1))

    def forward(self, inputs, lengths):
        # in: [sentence_length, batch_size]
        # sentence_length is the number of tokens composing the tweet (including paddings)
        x = self.embed(inputs)
        # out: [sentence_length, batch_size, EMBEDDING_DIM]
        # in this case, EMBEDDING_DIM is the number of features in input to the RNN

        # in: [sequence_length, batch_size, any_other_dimension]
        # Packs a Tensor containing padded sequences of variable length.
        x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), enforce_sorted=False)
        # out: a PackedSequence

        x, _hidden = self.recurrent_layer(x)
        # out: RNN layer produces two results:
        # - x: the output features from the last layer of the RNN
        #   - [sequence_length, batch_size, hidden_size]
        # - _hidden: the final hidden state for each element in the batch.
        #   - [1, batch_size, hidden_size]
        # NOTE: if the input was given in a PackedSequence, also the output will be a PackedSequence

        # Unpack the sequence. Returns the output of the RNN, unpacked.
        x, out_lengths = nn.utils.rnn.pad_packed_sequence(x)

        # Apply pooling to the features dimension produced and collapse them to a single value.
        # But first, a matrix transposition is necessary to align the result with the dimensions
        # required by the pooling layer
        # in: [batch_size, hidden_size, sequence_length]
        x = self.pooling(x.permute(1, 2, 0)).squeeze(-1)

        # in: [batch_size, hidden_dim]
        x = self.ffnn(x)
        # out: [batch_size, self.n_classes]
        # Ends with log_softmax
        return x


def load_dataset(cmd_args: Dict):
    batch_size = cmd_args.get("-b", BATCH_SIZE)
    embedding_dim = cmd_args.get("-emb_dim", EMBEDDING_DIM)
    train_file_size = cmd_args.get("-s", 0)
    return data.get_dataset(
        TRAIN_FILE[train_file_size],
        DEV_FILE,
        TEST_FILE,
        batch_size,
        batch_size,
        batch_size,
        embedding_dim)


def get_optimizer(model, cmd_args: Dict):
    lr = float(cmd_args.get("-lr", LR))
    optimizers = [optim.SGD(model.parameters(), lr=lr),
                  optim.Adamax(model.parameters(), lr=lr),
                  optim.Adam(model.parameters(), lr=lr)]
    return optimizers[cmd_args.get("-o", 1)]


def get_loss_function(cmd_args: Dict):
    loss_functions = [nn.NLLLoss(), nn.CrossEntropyLoss(), nn.MultiLabelMarginLoss()]
    return loss_functions[cmd_args.get("-l", 0)]


def get_model(cmd_args: Dict, glove_embeddings):
    return RNN1(glove_embeddings, cmd_args).to(DEVICE)


def training_step(model, optimizer, loss_function, loader, device):
    epoch_loss = 0
    epoch_acc = 0
    total = 0
    total_batches = (len(loader) + 1)

    # The following line only shuffles the dataset without sorting by lengths:
    # loader.dataset.shuffle()

    # This re-shuffles the dataset, and then sorts by sequence length.
    # It gives a bit on randomness on the sequences
    loader.dataset.shuffle_and_sort_by_length()

    model.train()
    for i, batch in enumerate(loader):
        lengths = batch["lengths"]
        x = batch["inputs"]
        y = batch["labels"]
        x, y, lengths = x.to(device), y.to(device), lengths.to(device)

        optimizer.zero_grad()
        y_pred = model(x, lengths)
        loss = loss_function(y_pred, y)

        loss.backward()
        optimizer.step()

        epoch_acc += (torch.argmax(y_pred, dim=1) == y).sum().item()
        epoch_loss += loss.item()
        total += len(y)

        if i % ((total_batches // 10) + 1) == 0:
            print(f"Progress {(i + 1) / total_batches:.2%}: Loss: {epoch_loss / total}, Acc: {epoch_acc / total:.2%}, "
                  f"correct: {epoch_acc}/{total}")

    accuracy = epoch_acc / total
    loss = epoch_loss / (len(loader) + 1)
    return accuracy, loss


def validation_step(model, loss_function, loader, device):
    total_loss = 0
    total_correct = 0
    total = 0
    total_batches = (len(loader) + 1)

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
            total += len(y)

    accuracy = total_correct / total
    loss = total_loss / total_batches
    return accuracy, loss


def main():
    # Read the command line arguments, if any, otherwise return the defaults
    cmd_args = utils.read_cmd_args(defaults={
        "-m": 1,
        "-l": 1,
        "-o": 1,
        "-lr": LR,
        "-h": 15,
        "-e": N_EPOCHS,
        "-rnn_l": 1,
        "-rrn_t": "lstm",
        "emb_dim": EMBEDDING_DIM,
        "-b": BATCH_SIZE,
    })
    train_loader, dev_loader, test_loader, glove_embeddings = load_dataset(cmd_args)
    model = get_model(cmd_args, glove_embeddings)
    optimizer = get_optimizer(model, cmd_args)
    loss_function = get_loss_function(cmd_args)
    epochs = cmd_args.get("-e")
    print("Command line arguments: ", cmd_args)

    history = []
    for epoch in range(epochs):
        print(f"\n\n--- Beginning Epoch {epoch} ---")
        train_acc, train_loss = training_step(model, optimizer, loss_function, train_loader, DEVICE)

        print("Validating...")
        valid_acc, valid_loss = validation_step(model, loss_function, dev_loader, DEVICE)
        history.append([epoch, train_acc, train_loss, valid_acc, valid_loss])

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    print(f"-- END TRAINING --")

    # Final Test
    print("** TESTING STEP **")
    test_accuracy, test_loss = validation_step(model, loss_function, test_loader, DEVICE)
    print(f"Final Accuracy: {test_accuracy:.2%}, Loss: {test_loss:.4f}")

    # Save the information on disk
    utils.save_results(history, test_accuracy, test_loss, cmd_args)


if __name__ == '__main__':
    main()
