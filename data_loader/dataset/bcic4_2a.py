import numpy as np
import torch
from torch.utils.data import Dataset
from random import uniform

class BCIC4_2A(Dataset):
    def __init__(self, args, phase):
        self.args = args
        self.load_data(phase)
        self.torch_form()
        self.reshape_data()

    def load_data(self, phase):
        if self.args.get_prediction:
            self.X = np.load(f"./data_loader/dataset/bcic4-2a/{phase}/S{self.args.subject:02}_X.npy")
        elif self.args.evaluation:
            self.y = np.load(f"./data_loader/dataset/bcic4-2a/label/S{self.args.subject:02}_y.npy")
        else:
            self.X = np.load(f"./data_loader/dataset/bcic4-2a/{phase}/S{self.args.subject:02}_X.npy")
            self.y = np.load(f"./data_loader/dataset/bcic4-2a/{phase}/S{self.args.subject:02}_y.npy")

            # flip_data = self.augmentation_flip()
            # flip_data = np.vstack([self.X, flip_data])
            # noise_data = self.augmentation_noise(flip_data)
            # self.X = np.vstack([flip_data, noise_data])
            # self.y = np.tile(self.y, 4)

            self.x1 = self.augmentation_noise()
            self.x2 = self.augmentation_flip()


    def torch_form(self):
        if self.args.get_prediction:
            self.X = torch.Tensor(self.X)
        elif self.args.evaluation:
            self.y = torch.LongTensor(self.y)
        else:
            self.X = torch.Tensor(self.X)
            self.y = torch.LongTensor(self.y)

    def reshape_data(self):
        if not self.args.evaluation:
            self.X = self.X

    def __len__(self):
        if not self.args.evaluation:
            return len(self.X)
        else:
            return len(self.y)

    def __getitem__(self, idx):
        if self.args.get_prediction:
            sample = self.X[idx]
            return sample
        elif self.args.evaluation:
            sample = self.y[idx]
            return sample
        else:
            # sample = [self.X[idx], self.y[idx]]
            sample = [self.x1[idx], self.x2[idx], self.X[idx], self.y[idx]]
            return sample
    
    def augmentation_flip(self):
        aug_list = []
        for data in self.X:
            data = np.asarray(data) # shape (22, 500)
            max_d = np.max(data)
            aug_list.append(max_d - data)
        aug_X = np.asarray(aug_list)
        return aug_X

    def augmentation_noise(self, c_noise=2):
        aug_list = []
        for data in self.X:
            data = np.asarray(data)
            sample_2D = []
            std_d = np.std(data)
            # print(std_d)
            rand = uniform(-0.5, 0.5)
            aug_list.append(data + rand * std_d / c_noise)
        aug_X = np.asarray(aug_list)
        return aug_X