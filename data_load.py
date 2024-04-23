import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import hdf5storage

def add_random_noise(eeg_data, noise_factor=0.01):
    noise = np.random.randn(*eeg_data.shape) * noise_factor
    augmented_data = eeg_data + noise
    return augmented_data
class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self.torch_form()

    def load_data(self):
        s = self.args.train_subject[0]
        if self.args.phase == 'train':
            # self.X = np.load(f"./data/S{s:02}_test_X.npy")
            # self.y = np.load(f"./answer/S{s:02}_y_test.npy")
            data = hdf5storage.loadmat(f"/home/easyonemain/Downloads/tmp2/ABA_proposal/xai611_mid_project-main/dataset/tr/sub_{s}tr.mat")
            self.X = data['x_data']
            self.y = data['y_label']
            # self.y = np.load(f"./data/S{s:02}_train_y.npy")
        elif self.args.phase == 'val':
            data = hdf5storage.loadmat(f"/home/easyonemain/Downloads/tmp2/ABA_proposal/xai611_mid_project-main/dataset/val/sub_{s}val.mat")
            self.X = data['x_data']
            self.y = data['y_label']
        elif self.args.phase == 'test':
            data = hdf5storage.loadmat(f"/home/easyonemain/Downloads/tmp2/ABA_proposal/xai611_mid_project-main/dataset/test/sub_{s}test.mat")
            self.X = data['x_data']
            self.y = data['y_label']
        self.X = np.transpose(self.X, (2, 1, 0))
        if len(self.X.shape) <= 3:
            self.X = np.expand_dims(self.X, axis=1)
        if self.args.CAR:
            print('CAR')
            self.X = self.X - np.mean(self.X, axis=2, keepdims=True)
        else:
            print('No CAR')
        # self.X = np.squeeze(self.X)
        if self.args.dataaug:
            print("data augmentation")
            auged_X = add_random_noise(self.X)
            self.X = np.concatenate((self.X, auged_X), axis=0)
            self.y = np.concatenate((self.y, self.y), axis=0)
        self.y = np.squeeze(self.y)
    def torch_form(self):
        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = [self.X[idx], self.y[idx]]
        return sample



def data_loader(args):
    print("[Load data]")
    # Load train data
    args.phase = "train"
    trainset = CustomDataset(args)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Load val data
    args.phase = "val"
    valset = CustomDataset(args)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    args.phase = "test"
    testset = CustomDataset(args)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    # Print
    print(f"train_set size: {train_loader.dataset.X.shape}")
    print(f"val_set size: {val_loader.dataset.X.shape}")
    print("")
    return train_loader, val_loader, test_loader
