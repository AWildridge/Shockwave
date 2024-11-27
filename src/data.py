import numpy as np
from numpy.random import default_rng
from scipy import fft
import torch
from torch.utils.data import Dataset

class MaskingDataset(Dataset):
    '''
    A torch dataset class that implements masking of a portion of each sample.
    It returns two samples for each item - the sample with a portion masked (input) and
    the same sample with nothing masked (target).
    '''
    def __init__(self, data, mask_portion = 0.1, mask_value=0.0, seed=None):
        '''

        '''
        super().__init__()

        self.dataset = data

        self._d_sample = self.dataset.size(1)
        self.mask_length = int(np.ceil(mask_portion * self._d_sample))
        self.max_start_index = self._d_sample - self.mask_length

        self.mask_value = mask_value

        self.gen = torch.Generator()
        if (seed is not None):
            self.gen.manual_seed(seed)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        start_index = torch.randint(0, self.max_start_index, [1], generator=self.gen).item()

        mask = torch.tensor([((i < start_index) or (i >= (start_index + self.mask_length))) for i in range(self._d_sample)], dtype=torch.bool)
        masked = torch.where(mask, self.dataset[idx], self.mask_value)

        return masked, self.dataset[idx]
    
    def __getitems__(self, idxs):
        start_indices = torch.randint(0, self.max_start_index, [len(idxs)], generator=self.gen)

        # Create a 2D mask where each row is a valid sample mask
        mask = torch.tensor([[((i < start_indices[j]) or (i >= (start_indices[j] + self.mask_length))) for i in range(self._d_sample)] for j in range(len(idxs))], dtype=torch.bool)

        batch = self.dataset[idxs]
        masked = torch.where(mask, batch, self.mask_value)

        return masked, batch

def train_valid_test_split(dataset_array, train_portion=0.7, valid_portion=0.15, test_portion=0.15, seed=None):
    '''
    Splits a dataset into training, validation, and testing sets according to the portions given (normalized to sum 1 if not already).
    Dataset must be given as a >=2D numpy array.
    NOTE: This function shuffles the given dataset in-place, meaning the order of samples in the dataset will be changed.
    '''
    total_portion = train_portion + valid_portion + test_portion
    if (total_portion != 1.0):
        train_portion /= total_portion
        valid_portion /= total_portion
        test_portion /= total_portion
    
    num_samples = len(dataset_array)

    if (seed is None):
        rand_gen = default_rng()
    else:
        rand_gen = default_rng(seed)
    
    # Shuffles dataset IN-PLACE
    rand_gen.shuffle(dataset_array)

    first_cut = int(train_portion * num_samples)
    second_cut = int((train_portion + valid_portion) * num_samples)

    train = dataset_array[:first_cut]
    valid = dataset_array[first_cut:second_cut]
    test = dataset_array[second_cut:]

    return train, valid, test

def PrepareDatasets(dataset, seed=None):
    '''
    Convenience function to take a large array of data and produce training, validation, and testing datasets.
    '''
    train_data, valid_data, test_data = train_valid_test_split(dataset, train_portion=0.8, valid_portion=0.1, test_portion=0.1, seed=seed)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_dataset = MaskingDataset(train_data, mask_portion=0.1, mask_value=-100.0)
    valid_dataset = MaskingDataset(valid_data, mask_portion=0.1, mask_value=-100.0)
    test_dataset = MaskingDataset(test_data, mask_portion=0.1, mask_value=-100.0)
    
    return train_dataset, valid_dataset, test_dataset

def PrepareIdentityDatasets(dataset, seed=None):
    train_data, valid_data, test_data = train_valid_test_split(dataset, train_portion=0.8, valid_portion=0.1, test_portion=0.1, seed=seed)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    valid_data = torch.tensor(valid_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
    valid_dataset = torch.utils.data.TensorDataset(valid_data, valid_data)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_data)

    return train_dataset, valid_dataset, test_dataset

def PrepareTwoChannelSimpleDatasets(dataset, seed=None):
    train_data, valid_data, test_data = train_valid_test_split(dataset, train_portion=0.8, valid_portion=0.1, test_portion=0.1, seed=seed)

    train_inputs = train_data[:, 0, :]
    train_targets = train_data[:, 1, :]

    valid_inputs = valid_data[:, 0, :]
    valid_targets = valid_data[:, 1, :]

    test_inputs = test_data[:, 0, :]
    test_targets = test_data[:, 1, :]

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(valid_inputs, dtype=torch.float32), torch.tensor(valid_targets, dtype=torch.float32))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32))

    return train_dataset, valid_dataset, test_dataset

def PrepareFFTDatasets(dataset, seed=None):
    train_data, valid_data, test_data = train_valid_test_split(dataset, train_portion=0.8, valid_portion=0.1, test_portion=0.1, seed=seed)

    train_inputs = torch.tensor(train_data, dtype=torch.float32)
    train_targets = torch.tensor(np.abs(fft.rfft(train_data)[:, :, :-1]), dtype=torch.float32)

    valid_inputs = torch.tensor(valid_data, dtype=torch.float32)
    valid_targets = torch.tensor(np.abs(fft.rfft(valid_data)[:, :, :-1]), dtype=torch.float32)

    test_inputs = torch.tensor(test_data, dtype=torch.float32)
    test_targets = torch.tensor(np.abs(fft.rfft(test_data)[:, :, :-1]), dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    valid_dataset = torch.utils.data.TensorDataset(valid_inputs, valid_targets)
    test_dataset = torch.utils.data.TensorDataset(test_inputs, test_targets)

    return train_dataset, valid_dataset, test_dataset