


import torch
import torchvision
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, RandomHorizontalFlip, Normalize, ToPILImage, RandomCrop
import csv
import os
import numpy as np
from sklearn.model_selection import train_test_split
from option.config import Config
from PIL import Image
import scipy.misc as m
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


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
    def __init__(self, csv_path, data_path, scale_1, img_size, transforms=None, crop_size=None, is_train=True):
        super().__init__()
        self.transforms = transforms
        self.is_train = is_train
        self.p = 0.5
        self.img_size = img_size
        self.csv_path = csv_path
        self.data_path = data_path
        self.crop_size = crop_size
        self.scale_1 = scale_1

        self.data_tmp, self.dist_img_path, self.dist_type, self.dmos = self.csv_read(self.csv_path)

        self.x_train, self.x_test, self.y_train, self.y_test = self.train_test_split(self.data_tmp, self.dmos, test_size=0.2, random_state=2, shuffle=True)

    def __getitem__(self, idx):
        # if (self.is_train):
        #     if (self.crop_size is not None) and (self.img_size is None):
        #         dist_img = self.img_read(self.data_path, self.x_train[idx][0])
        #         dist_img = self.to_tensor(dist_img)
        #         dist_img = self._normalize(dist_img)
        #         dist_img = self._horizontal_flip(dist_img)
        #         dist_img = self._vertical_flip(dist_img)
        #         dist_img = self._random_crop(dist_img, crop_size=self.crop_size)
        #         return dist_img, self.y_train[idx]
        #     elif (self.crop_size is None) and (self.img_size is not None):
        #         dist_img = self.img_read(self.data_path, self.x_train[idx][0])
        #         dist_img = self.re_size(dist_img, self.img_size)
        #         dist_img = self.to_tensor(dist_img)
        #         dist_img = self._normalize(dist_img)
        #         dist_img = self._horizontal_flip(dist_img)
        #         dist_img = self._vertical_flip(dist_img)
        #         return dist_img, self.y_train[idx]
        # else:
        if (self.is_train):
            dist_img = self.img_read(self.data_path, self.x_train[idx][0])
            dist_img = self.re_size(dist_img)
            return dist_img, self.y_train[idx], self.x_train[idx][0]
        else:
            dist_img = self.img_read(self.data_path, self.x_test[idx][0])
            dist_img = self.re_size(dist_img)
            return dist_img, self.y_test[idx], self.x_test[idx][0]


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
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            next(reader)
            # for row in reader:
            for row in tqdm(reader):
                data_tmp.append(row[0:3])
                dist_img_path.append(row[0])
                dist_type.append(row[1])
                dmos.append(float(row[3]))
        return data_tmp, dist_img_path, dist_type, dmos


    def re_size(self, dist_img):
        # dist_img = cv2.resize(dist_img, (img_size, img_size))
        # return dist_img
        h, w, c = dist_img.shape
        d_img_scale_1 = cv2.resize(dist_img, dsize=(self.scale_1, int(h * (self.scale_1 / w))),
                                   interpolation=cv2.INTER_CUBIC)

        return d_img_scale_1

    def img_read(self, data_path, dist):
        # dist = dist[dist.rfind('/')+1:]

        distt = os.path.join(data_path + dist)
        dist_img = cv2.imread(distt, cv2.IMREAD_COLOR)
        dist_img = cv2.cvtColor(dist_img, cv2.COLOR_BGR2RGB)
        dist_img = np.array(dist_img).astype('float32')/255
        return dist_img

    def to_tensor(self, img):
        totensor = ToTensor()
        img_t = totensor(img)
        return img_t

    def mosMinMaxNorm(self, mos):
        mos_norm = []
        mos_max = max(mos)
        mos_min = min(mos)
        for idx in range(len(mos)):
            mos_norm.append(float(1. - (mos[idx] - mos_min) / (mos_max - mos_min)))
        return mos_norm

    def load_image(self, data):
        img = cv2.imread(data)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _normalize(self, x):
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        x = normalize(x)
        return x

    def _horizontal_flip(self, x):
        if torch.rand(1) < 0.5:
            return F.hflip(x)
        return x

    def _vertical_flip(self, x):
        if torch.rand(1) < 0.5:
            return F.vflip(x)
        return x

    def _random_rotate(self, x):
        degrees = [45, 90]
        if torch.rand(1) < 0.5:
            idx = np.random.randint(0, 2)
            x = F.rotate(x, degrees[idx])
        return x

    def _random_crop(self, x, crop_size):
        randomcrop = RandomCrop(crop_size)
        i, j, h, w = randomcrop.get_params(x, output_size=(crop_size, crop_size))
        x = F.crop(x, i, j, h, w)
        return x


if __name__ == "__main__":
    csv_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\NRIQA\\all_data_csv\\KADID-10k.csv'
    data_path = 'C:\\Users\\yunjeongyong\\Desktop\\intern\\Triq-yunjeong\\data\\'
    dataset = KadidDataset(csv_path, data_path, scale_1=384, scale_2=224, img_size=None, transforms=None, is_train=True)


    for i in range(10):
        dist_img, scale1, scale2, dmos = dataset[i]
        # plt.imshow(torchvision.utils.make_grid(dist_img, nrow=5).permute(1, 2, 0))
        # dist_img = dist_img.swapaxes(0,1)
        # dist_img = dist_img.swapaxes(1,2)
        plt.imshow(dist_img)
        plt.show()
        print(dist_img)
    print(0)
