import pandas as pd
import torch
from torchtext.vocab import GloVe, vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def _get_data(file_path: str):
    print(f'Loading {file_path}')
    dataset = pd.read_csv(file_path)
    dataset.columns = ["label", "tweetid", "timestamp", "query", "user", "tweet"]
    return (
        dataset["tweet"].tolist(),
        dataset["label"].tolist(),
    )


def collate_fn(data):
    tweets = [d["tweet"] for d in data]
    labels = [d["label"] for d in data]
    lengths = torch.tensor([len(d["tweet"]) for d in data])

    inputs = pad_sequence(tweets)
    labels = torch.tensor(labels)
    return {
        "inputs": inputs,
        "labels": labels,
        "lengths": lengths
    }


class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, tweets, labels, tokenizer, glove_vocab):
        self.tokenizer = tokenizer
        self.tweets = tweets
        self.labels = labels
        self.glove_vocab = glove_vocab

        # global_vectors = GloVe(name='twitter.27B', dim=embedding_dim)

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.tweets.items()}
        # item = {key: torch.tensor(global_vectors.get_vecs_by_tokens(self.tokenizer(val[idx]), lower_case_backup=True)) for key, val in self.tweets.items()}
        item = dict()
        item["tweet"] = torch.tensor(self.glove_vocab(self.tokenizer(self.tweets[idx])))
        item["label"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.tweets)


def get_dataset(tokenizer, train_file, dev_file, test_file, train_batch_size, dev_batch_size, test_batch_size,
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

    print("Getting datasets")
    train_dataset = TwitterDataset(train_tweets, train_labels, tokenizer, glove_vocab)
    dev_dataset = TwitterDataset(dev_tweets, dev_labels, tokenizer, glove_vocab)
    test_dataset = TwitterDataset(test_tweets, test_labels, tokenizer, glove_vocab)

    print("Getting loaders")
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, collate_fn=collate_fn, shuffle=False)

    return train_loader, dev_loader, test_loader, glove_embeddings
