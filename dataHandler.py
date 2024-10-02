from torch.utils.data import Dataset
from torch import tensor

class TranslationDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return tensor(self.data['padded_eng'].iloc[index][1:]), tensor(self.data['padded_fr'].iloc[index])
