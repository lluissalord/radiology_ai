import numpy as np

from sklearn.model_selection import train_test_split
import PIL
from tqdm import tqdm

from torch.utils.data import Dataset

from fastai.data.core import TfmdDL

class AllLabelsInBatchDL(TfmdDL):
    """ DataLoader which allows to have a minimum of samples of all the labels in each batch """
    def __init__(self, dataset=None, min_samples=1, **kwargs):
        super().__init__(dataset=dataset, **kwargs)
        if self.bs < len(self.vocab):
            print('AllLabelsInBatchDL working as simple DL because batch size is less than number of labels')
            self.min_samples = 0
        else:
            self.min_samples = min_samples

    def get_idxs(self):
        if self.n==0: return []
        idxs = super().get_idxs()
        if not self.shuffle: return idxs

        # Transform to numpy array to replace efficiently
        idxs = np.array(idxs)

        # Generate random indexes which will be substituted by the labels
        n_batches = self.n // self.bs
        idxs_subs = [np.random.choice(self.bs, len(self.vocab) * self.min_samples, replace=False) + i * self.bs for i in range(n_batches)]

        # Iterate along batches and substitute selected indexes with label indexes
        for batch_idxs_subs in idxs_subs:
            label_idxs = []
            for label in self.vocab:
                # Extract indexes of current label and randomly choose `min_samples`
                label_idx = list(self.items[self.col_reader[1](self.items) == label].index)
                label_idx = list(np.random.choice(label_idx, size=self.min_samples, replace=True))

                label_idxs = label_idxs + label_idx
            
            # Shuffle label indexes and replace them
            np.random.shuffle(label_idxs)
            idxs[batch_idxs_subs] = label_idxs
        
        return idxs


class SelfSupervisedDataset(Dataset):
    def __init__(self, df, validation = False, transform=None, path_col='Original_Filename', prefix='', suffix='.png'):

        self.transform = transform

        #use sklearn's module to return training data and test data
        if validation:
            _, self.df = train_test_split(df, test_size=0.20, random_state=42)

        else:
            self.df, _ = train_test_split(df, test_size=0.20, random_state=42)

        self.image_pairs = []

        for idx, d in tqdm(enumerate(self.df[path_col]), total=len(self.df.index)):
          
            im = PIL.Image.open(prefix + d + suffix).convert('RGB')

            if self.transform:
                sample = self.transform(im) #applies the SIMCLR transform required, including new rotation
            else:
                sample = im

            self.image_pairs.append(sample)
          
    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        #doing the PIL.image.open and transform stuff here is quite slow
        return (self.image_pairs[idx], 0)