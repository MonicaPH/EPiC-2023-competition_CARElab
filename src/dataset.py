import torch
from torch.utils.data import Dataset
import numpy as np


class EpicDataset(Dataset):
    """Usage example:
    from dataloader import S1
    from dataset import EpicDataset
    trainScenario1Dataset = EpicDataset(S1())
    testScenario1Dataset = EpicDataset(S1(), is_test=True)
    """
    def __init__(self, scenario, modality, windowLen=50, is_test=False):
        self.is_test = is_test
        self.loader = scenario
        self.data = np.zeros((windowLen,1))
        self.label = np.zeros((1,2))
        self.numBlocks = 0
        for sub, vid in [
            x[(not self.is_test)*'train'+self.is_test*'test']
            for x in scenario.train_test_groups()
        ]:
            data, label = self.loader.train_data(sub, vid)
            numBlocks = len(data) // windowLen
            endIdx = windowLen * numBlocks
            # rearrange data and annotations
            self.data = np.concatenate([
                self.data,
                data[modality].values[:endIdx].reshape(windowLen, -1)
            ], axis=1)
            self.label = np.concatenate([
                self.label,
                label.values[len(label)-numBlocks:]
            ])
            self.numBlocks += numBlocks
        # transform to tensor
        self.data = torch.Tensor(self.data)
        self.label = torch.Tensor(self.label)

    def __len__(self):
        return self.numBlocks-1 # 1 dummy entry at the beginning to concatenate with 0's

    def __getitem__(self, idx):
        return self.data[:, 1+idx], self.label[1+idx, :]