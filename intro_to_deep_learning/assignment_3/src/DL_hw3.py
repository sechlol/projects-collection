# -*- coding: utf-8 -*-
"""
   Introduction to Deep Learning
   Assignment 3: Sentiment Classification of Tweets on a Recurrent Neural Network using Pretrained Embeddings
   Hande Celikkanat
   Credit: Data preparation pipeline adopted from https://medium.com/@sonicboom8/sentiment-analysis-torchtext-55fb57b1fab8
"""

import sys
from typing import Dict, Any

import regex as re
import spacy
import torch
import torch.nn as nn
import torch.optim as optim

import data
import utils

# Constants - Add here as you wish
N_EPOCHS = 5
LR = 0.01
EMBEDDING_DIM = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_FILE = '../data/sent140.train.mini.csv'
DEV_FILE = '../data/sent140.dev.csv'
TEST_FILE = '../data/sent140.test.csv'

TRAIN_BS = 32
DEV_BS = 32
TEST_BS = 32

re1 = re.compile(r'[^A-Za-z0-9]+')
re2 = re.compile(r'https?:/\/\S+')

# Auxilary functions for data preparation
tok = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', "lemmatizer"])


def tokenizer(s):
    return [w.text.lower() for w in tok(tweet_clean(s))]


def tweet_clean(text):
    text = re1.sub(' ', text)  # remove non alphanumeric character
    text = re2.sub(' ', text)  # remove links
    return text.strip()


def read_cmd_args(defaults: Dict = None) -> Dict[str, Any]:
    args = defaults or {}
    for i in range(1, len(sys.argv), 2):
        # check if the argument has a corresponding value
        if i + 1 < len(sys.argv):
            # add the argument and its value to the dictionary
            args[sys.argv[i]] = int(sys.argv[i + 1])
    return args


# Utility
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def load_dataset():
    return data.get_dataset(
        tokenizer,
        TRAIN_FILE,
        DEV_FILE,
        TEST_FILE,
        TRAIN_BS,
        DEV_BS,
        TEST_BS,
        EMBEDDING_DIM)


def get_optimizer(model, cmd_args: Dict):
    lr = float(cmd_args.get("-lr", LR))
    optimizers = [optim.SGD(model.parameters(), lr=lr),
                  optim.Adamax(model.parameters(), lr=lr),
                  optim.Adam(model.parameters(), lr=lr)]
    return optimizers[cmd_args.get("-o", 1)]


def get_loss_function(cmd_args: Dict):
    loss_functions = [nn.NLLLoss(), nn.CrossEntropyLoss(), nn.MultiLabelMarginLoss()]
    return loss_functions[cmd_args.get("-l", 1)]


# Recurrent Network
class RNN1(nn.Module):
    def __init__(self, embedding_matrix, cmd_args):
        super(RNN1, self).__init__()
        hidden_size = cmd_args.get("-h", 15)
        n_layers = cmd_args.get("-rnn_l", 1)
        use_tanh = "-rnn_tanh" in cmd_args

        self.embed = nn.Embedding.from_pretrained(embedding_matrix)
        if "-lstm" in cmd_args:
            self.recurrent_layer = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=hidden_size)
        else:
            self.recurrent_layer = nn.RNN(
                input_size=embedding_matrix.size(1),
                hidden_size=hidden_size,
                num_layers=n_layers,
                nonlinearity="tanh" if use_tanh else "relu")

        # NOTE: Pooling kernel is big as the whole hidden size: it will collapse the dimension
        # to a single value
        self.pooling = nn.AdaptiveMaxPool1d(output_size=1)
        self.ffnn = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size, out_features=2),
            nn.ReLU(),
            nn.LogSoftmax(dim=1))

    def forward(self, inputs, lengths):
        # in: [sentence_length, batch_size]
        # sentence_length is the number of tokens composing the tweet
        x = self.embed(inputs)

        # packing
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)

        # out: [sentence_length, batch_size, EMBEDDING_DIM]
        # in this case, EMBEDDING_DIM is the number of features in input to the RNN
        x, _hidden = self.recurrent_layer(x)
        # out: RNN layer produces two results:
        # - x: the output features from the last layer of the RNN
        #   - [sequence_length, batch_size, hidden_size]
        # - _hidden: the final hidden state for each element in the batch.
        #   - [1, batch_size, hidden_size]

        # padding
        x, _ = nn.utils.rnn.pad_packed_sequence(x)

        # Apply pooling to the extra dimension produced by the RNN. It will be collapsed
        # to a single value. But first, a matrix transposition is necessary to align the result with
        # The dimensions required by the pooling layer
        # TODO: Do pooling
        # x = torch.transpose(torch.transpose(x, 0, 1), 1, 2)
        # x = x.permute(1, 0, 2)
        # in: [batch_size, hidden_size, sequence_length]
        x = self.pooling(x.permute(1, 2, 0)).squeeze(-1)
        # x = x[-1]

        # in: [batch_size, hidden_dim]
        x = self.ffnn(x)
        # out: [batch_size, self.n_classes]
        # Ends with log_softmax
        return x


class RNN2(nn.Module):
    def __init__(self, embedding_matrix, cmd_args):
        super(RNN2, self).__init__()
        hidden_size = cmd_args.get("-h", 15)
        n_layers = cmd_args.get("-rnn_l", 1)
        use_lstm = "-lstm" in cmd_args
        use_tanh = "-rnn_tanh" in cmd_args

        if use_lstm:
            self.recurrent_layer = nn.LSTM(input_size=embedding_matrix.shape[1], hidden_size=hidden_size)
        else:
            self.recurrent_layer = nn.RNN(
                input_size=embedding_matrix.size(1),
                hidden_size=hidden_size,
                num_layers=n_layers,
                nonlinearity="tanh" if use_tanh else "relu")
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=False)
        x, _ = self.recurrent_layer(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        x = self.pooling(x.permute(1, 2, 0)).squeeze(-1)
        x = self.fc(x)
        return nn.functional.log_softmax(x, dim=1)


def get_model(cmd_args: Dict, glove_embeddings):
    m = cmd_args.get("-m", 0)
    if m == 0:
        model = RNN1(glove_embeddings, cmd_args)
    else:
        model = RNN2(glove_embeddings, cmd_args)
    return model.to(DEVICE)


def main():
    train_loader, dev_loader, test_loader, glove_embeddings = load_dataset()
    cmd_args = read_cmd_args(defaults={})
    model = get_model(cmd_args, glove_embeddings)
    optimizer = get_optimizer(model, cmd_args)
    loss_function = get_loss_function(cmd_args)
    epochs = cmd_args.get("-e", N_EPOCHS)

    history = []
    for epoch in range(epochs):
        print(f"\n\n--- Beginning Epoch {epoch} ---")
        train_loss, train_acc = utils.training_step(model, optimizer, loss_function, train_loader, DEVICE)

        print("Validating...")
        valid_loss, valid_acc = utils.validation_step(model, loss_function, dev_loader, DEVICE)
        history.append([epoch, train_acc, train_loss, valid_acc, valid_loss])

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    print(f"-- END TRAINING --")
    utils.save_history(history)

    # Final Test
    print("** TESTING STEP **")
    test_accuracy, test_loss = utils.validation_step(model, loss_function, test_loader, DEVICE)
    print(f"Final Accuracy: {test_accuracy:.2%}, Loss: {test_loss:.4f}")
    utils.save_test_result(test_accuracy, test_loss)


if __name__ == '__main__':
    main()
