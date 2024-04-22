import random
import pandas as pd
import torch
import os
from torch.utils.data import Dataset


class MEDataset(Dataset):
    AMP_LIST = [1.2, 1.4, 1.6, 1.8, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0]

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict, catego: str,
                 train: bool, parallel=None, mat_dir=None):

        self.data_info = data_info
        self.label_mapping = label_mapping
        self.catego = catego
        self.train = train
        self.parallel = parallel
        self.mat_dir = mat_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        # Label for the image
        label = self.label_mapping[self.data_info.loc[idx, "Estimated Emotion"]]

        subject = self.data_info.loc[idx, "Subject"]
        folder = self.data_info.loc[idx, "Filename"]

        if self.parallel:
            n_patches = torch.empty(self.parallel, 30, 7, 7)
            for i in range(self.parallel):
                if self.train:
                    amp_factor = random.choice(MEDataset.AMP_LIST)
                    mat_dir = f"{self.mat_dir}\\Inter_offset_{self.parallel}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
                    while not os.path.exists(mat_dir):
                        amp_factor = random.choice(MEDataset.AMP_LIST)
                        mat_dir = f"{self.mat_dir}\\Inter_offset_{self.parallel}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
                else:
                    if folder == "s3_sur_01":
                        amp_factor = 1.8
                    else:
                        amp_factor = 2.0
                # onset to apex
                # mat_dir = f"{self.mat_dir}\\Inter_{self.parallel}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
                # onset to offset
                mat_dir = f"{self.mat_dir}\\Inter_offset_{self.parallel}\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"

                n_patches[i] = torch.load(mat_dir)[i]
        else:
            if self.train:
                amp_factor = random.choice(MEDataset.AMP_LIST)
            else:
                amp_factor = 2.0

            if self.catego == "SAMM":
                pass
            elif self.catego == "Cropped":
                pass
            else:
                mat_dir = f"{self.mat_dir}\\Inter_1\\sub{subject.zfill(2)}_{folder}_{amp_factor}.pt"
            n_patches = torch.load(mat_dir)

        return n_patches, label
