from torch.utils.data import Dataset
import os
import pickle
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2



#  data in one .pkl file, for small size dataset
class TOF_real(Dataset):    # real data
    def __init__(self, TrainNoisy_path, TrainLabel_path, training=True):
        self.training = training
        f_t = open(TrainNoisy_path, 'rb')
        self.data_t = pickle.load(f_t)
        f_t.close()
        self.train = torch.from_numpy(np.stack(self.data_t, axis=0)).float()   # 6 channels: [raw0, raw1, raw2, raw3, Q, I]

        f_l = open(TrainLabel_path, 'rb') 
        self.data_l = pickle.load(f_l)
        f_l.close()
        self.label = torch.from_numpy(np.stack(self.data_l, axis=0)).float()   # 7 channels: [raw0, raw1, raw2, raw3, Q, I, depth]

        self.length = len(self.data_t[0])
        print("dataset has been loaded into mem already....")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.training:
            return self.train[:, idx, 2:-2, :], self.label[:, idx, 2:-2, :]     # img1, img2
        else:
            return self.train[:, idx,...], self.label[:, idx,...]      # eval


class ToF_synthetic(Dataset):   # synthetic data
    def __init__(self, Data_path, training=True):

        self.DataTransform = A.Compose([
            # transforms.ToPILImage(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-3, 3)),
            ToTensorV2(transpose_mask=False)
        ])

        self.training = training
        self.Datapath = Data_path
        self.raw_data_list = sorted(os.listdir(self.Datapath))

    def __len__(self):
        return len(self.raw_data_list)

    def __getitem__(self, idx):
        raw_idx = self.raw_data_list[idx]    # acquire raw data according to index
        raw_path = os.path.join(self.Datapath, raw_idx)
        self.raw = np.fromfile(raw_path, dtype=np.float32).reshape([10, 180, 240])
        self.raw_training = self.raw.transpose(1, 2, 0)

        if self.training:
            self.train_data = self.DataTransform(image=self.raw_training)['image']
            self.label = self.train_data[0:5, ...]
            self.train = self.train_data[5:10, ...]

            return self.train[:, 2:-2, :], self.label[:, 2:-2, :]
        else:
            self.train_data = torch.from_numpy(self.raw)
            self.label = self.train_data[0:5, ...]
            self.train = self.train_data[5:10, ...]     # torch.Size([5, 180, 240])

            return self.train, self.label       # img1, img2