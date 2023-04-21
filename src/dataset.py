from torch.utils.data import Dataset


class EpicDataset(Dataset):
    """Usage example:
    from dataloader import S1
    from dataset import EpicDataset
    trainScenario1Dataset = EpicDataset(S1())
    testScenario1Dataset = EpicDataset(S1(), is_test=True)
    """
    def __init__(self, szenario, is_test=False):
        self.is_test = is_test
        self.loader = szenario
        self.groups = self.loader.train_test_groups()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        data, label = self.loader.train_data(
            *self.groups[idx].get((not self.is_test)*'train'+self.is_test*'test')
        )
        return data, label