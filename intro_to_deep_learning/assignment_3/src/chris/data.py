import pandas as pd
import regex as re
import spacy
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe, vocab

re1 = re.compile(r'[^A-Za-z0-9]+')
re2 = re.compile(r'https?:/\/\S+')

# Auxilary functions for data preparation
tok = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner', "lemmatizer"])


def _get_data(file_path: str):
    print(f'Loading {file_path}')
    dataset = pd.read_csv(file_path)
    # dataset = pd.read_csv(file_path, nrows=100)
    dataset.columns = ["label", "tweetid", "timestamp", "query", "user", "tweet"]
    return (
        dataset["tweet"].tolist(),
        dataset["label"].tolist(),
    )


def _tokenizer(s):
    return [w.text.lower() for w in tok(_tweet_clean(s))]


def _tweet_clean(text):
    text = re1.sub(' ', text)  # remove non alphanumeric character
    text = re2.sub(' ', text)  # remove links
    return text.strip()


def collate_fn(data):
    tweets = [d["tweet"] for d in data]
    labels = [d["label"] for d in data]
    lengths = [d["length"] for d in data]

    inputs = pad_sequence(tweets)
    labels = torch.tensor(labels)
    lengths = torch.tensor(lengths)
    return {
        "inputs": inputs,
        "labels": labels,
        "lengths": lengths
    }


class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, labels, tokenizer, glove_vocab):
        self.labels = torch.tensor(labels)
        self.tweets_tokenized = [
            torch.tensor(glove_vocab(tokenizer(tweet)))
            for tweet in tweets]

        self.lengths = torch.tensor([len(seq) for seq in self.tweets_tokenized], dtype=torch.int)
        self.ids = torch.arange(len(self))
        self.sorted_ids = torch.argsort(self.lengths[self.ids], descending=True)

    def shuffle_and_sort_by_length(self):
        self.ids = torch.randperm(len(self))
        self.sorted_ids = torch.argsort(self.lengths[self.ids], descending=True)

    def shuffle(self):
        self.sorted_ids = torch.randperm(len(self))

    def __getitem__(self, idx):
        sorted_idx = self.ids[self.sorted_ids[idx]]
        return {
            "tweet": self.tweets_tokenized[sorted_idx],
            "label": self.labels[sorted_idx],
            "length": self.lengths[sorted_idx]
        }

    def __len__(self):
        return len(self.labels)


def get_dataset(train_file, dev_file, test_file, train_batch_size, dev_batch_size, test_batch_size,
                embedding_dim):
    train_tweets, train_labels = _get_data(train_file)
    dev_tweets, dev_labels = _get_data(dev_file)
    test_tweets, test_labels = _get_data(test_file)

    unk_index = 0
    glove_vectors = GloVe(name='twitter.27B', dim=embedding_dim)
    glove_vocab = vocab(glove_vectors.stoi)
    glove_vocab.insert_token("<unk>", unk_index)
    glove_vocab.set_default_index(unk_index)

    glove_embeddings = glove_vectors.vectors
    glove_embeddings = torch.cat((torch.zeros(1, glove_embeddings.shape[1]), glove_embeddings))

    print("Loading Dev Dataset...")
    dev_dataset = TwitterDataset(dev_tweets, dev_labels, _tokenizer, glove_vocab)
    print("Loading Test Dataset...")
    test_dataset = TwitterDataset(test_tweets, test_labels, _tokenizer, glove_vocab)
    print("Loading Train Dataset...")
    train_dataset = TwitterDataset(train_tweets, train_labels, _tokenizer, glove_vocab)

    print("Allocating loaders...")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=False)
    dev_loader = DataLoader(dev_dataset, batch_size=dev_batch_size, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, dev_loader, test_loader, glove_embeddings
