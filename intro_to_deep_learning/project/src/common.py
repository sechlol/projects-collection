import torch

LABELS = ['baby', 'bird', 'car', 'clouds', 'dog', 'female', 'flower', 'male', 'night', 'people', 'portrait', 'river', 'sea', 'tree']
LABELS_ENCODED = dict(zip(LABELS, torch.nn.functional.one_hot(torch.arange(len(LABELS)))))
