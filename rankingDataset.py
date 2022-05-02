import torch

class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings1, encodings2, labels, user_id):
        self.encodings1 = encodings1
        self.encodings2 = encodings2
        self.labels = labels
        self.user_id = user_id

    def __getitem__(self, idx):
        item1 = {key + "_1": torch.tensor(val[idx]) for key, val in self.encodings1.items()}
        item2 = {key + "_2": torch.tensor(val[idx]) for key, val in self.encodings2.items()}
        item = dict(**item1, **item2)
        item['labels'] = torch.tensor(self.labels[idx])
        item['user_id'] = torch.tensor(self.user_id[idx])
        return item

    def __len__(self):
        return len(self.labels)