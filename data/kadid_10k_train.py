


import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, Normalize, ToPILImage, RandomCrop
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from config import Config
from PIL import Image
import scipy.misc as m
import cv2
from tqdm import tqdm
from skimage.io import imread
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from piq import brisque, fsim
from torchvision.transforms import ToTensor, ToPILImage


'''
1, 이미지를 불러온다.
2. 텐서로 바꿈
3. normalize
4. augmentation
 4-1. random horizontal flip
 4-2. random vertical flip
 4-3. random rotation 
5. 랜덤 크롭 (256 x 256) <-> 혹은 resize?
'''
config = Config()

class KadidDataset(Dataset):
    def __init__(self, csv_path, data_path, is_train=True):
        super().__init__()
        self.is_train = is_train
        self.csv_path = csv_path
        self.data_path = data_path
        self.data_tmp, self.dist_img_path, self.dist_type, self.ref_img, self.dmos = self.csv_read(self.csv_path)
        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(self.data_tmp, self.dmos, test_size=0.2, random_state=2, shuffle=True)


    def __getitem__(self, idx):
        if (self.is_train):
            if self.crop_size is not None:
                dist_img, ref_img = self.img_read(self.data_path, self.x_train[idx][0], self.x_train[idx][2])
                return dist_img, ref_img, self.y_train[idx]

        else:
            dist_img, ref_img = self.img_read(self.data_path, self.x_test[idx][0], self.x_test[idx][2])

            return dist_img, ref_img, self.y_test[idx]

    def __len__(self):
        if self.is_train:
            return len(self.y_train)
        else:
            return len(self.y_test)

    def train_test_split(self, X, y, test_size, random_state, shuffle):
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
        )
        return x_train, x_test, y_train, y_test


    def csv_read(self, csv_path):
        dist_img_path = []
        dist_type = []
        dmos = []
        data_tmp = []
        ref_img = []
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            # for row in reader:
            for row in tqdm(reader):
                data_tmp.append(row[0:3])
                dist_img_path.append(row[0])
                dist_type.append(row[1])
                ref_img.append(row[2])
                dmos.append(float(row[3]))

        return data_tmp, dist_img_path, dist_type, ref_img, dmos

    def img_read(self, data_path, dist, ref):
        # dist = dist[dist.rfind('/')+1:]
        disttensor = imread(data_path + dist)
        disttensor = torch.Tensor(disttensor).permute(2,0,1)[None, ...]/255.
        # distt = os.path.join(data_path + dist)

        reftensor = imread(data_path + ref)
        reftensor = torch.Tensor(reftensor).permute(2, 0, 1)[None, ...] / 255.
        # dist_img = cv2.imread(distt, cv2.IMREAD_COLOR)
        # dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
        # dist_img = np.array(dist_img).astype('float32') / 255
        #
        # reff = os.path.join(data_path + ref)
        # ref_img = cv2.imread(reff, cv2.IMREAD_COLOR)
        # ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
        # ref_img = np.array(ref_img).astype('float32') / 255
        # # dist_img = np.transpose(dist_img, (2, 0, 1))

        return disttensor, reftensor

    def iter(self):
        for idx in range(self.__len__()):
            yield self.__getitem__(idx)



if __name__ == "__main__":
    csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\NRIQA\\all_data_csv\\KADID-10k.csv'
    data_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\'

    testset = KadidDataset(csv_path, data_path, is_train=False)
    testloader = DataLoader(testset,
                            batch_size=4,
                            shuffle=False,
                            drop_last=True,
                            num_workers=0,
                            pin_memory=False)


    brisque_array = []
    fsim_array = []
    dmos_array = []

    for idx, (dist, ref, dmos) in enumerate(testloader):
        brisque_array.append(brisque(dist, data_range=1., reduction='none'))
        fsim_array.append(fsim(dist, ref, data_range=1., reduction='none'))
        dmos_array.append(dmos)

    fsim_array = np.array(fsim_array)
    brisque_array = np.array(brisque_array)



    def minmax_normalize(x, min, max):
        return (x - min) / (max - min)

    def normalized(array):
        _min = min(array)
        _max = max(array)
        return [minmax_normalize(x, _min, _max) for x in array]

    fsim_array = normalized(fsim_array)
    brisque_array = normalized(brisque_array)

    fsim_result_s, _ = scipy.stats.spearmanr(fsim_array, dmos_array)
    fsim_result_p, _ = scipy.stats.pearsonr(fsim_array, dmos_array)
    print('SROCC:%4f / PLCC:%4f' % (fsim_result_s, fsim_result_p))
    brisque_result_s, _ = scipy.stats.spearmanr(brisque_array, dmos_array)
    brisque_result_p, _ = scipy.stats.pearsonr(brisque_array, dmos_array)
    print('SROCC:%4f / PLCC:%4f' % (brisque_result_s, brisque_result_p))


    for i in range(10):
        dist_img, dmos = dataset[i]
        # plt.imshow(torchvision.utils.make_grid(dist_img, nrow=5).permute(1, 2, 0))
        dist_img = dist_img.swapaxes(0,1)
        dist_img = dist_img.swapaxes(1,2)
        # plt.imshow(dist_img)
        # plt.show()
        print(dist_img)
        print(csv_path)
        print(dmos)

    print(0)
