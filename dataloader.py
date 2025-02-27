"""
Author : Swapnil Bembde
"""

import os
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


def data_split(dir):
    data = pd.read_csv(dir)
    df = pd.DataFrame(data)
    labels = df['Label']
    images = df['Filename']
    labels_train, labels_test, images_train, images_test = train_test_split(labels, images, test_size=0.3, random_state=5)

    df1 = pd.DataFrame(labels_train)
    df1 = pd.concat([df1,images_train],axis = 1)
    df1.to_csv('train_splt.csv', sep=',', encoding='utf-8',index=False)

    df2 = pd.DataFrame(labels_test)
    df2 = pd.concat([df2,images_test],axis = 1)
    df2.to_csv('test_splt.csv', sep=',', encoding='utf-8',index=False)
 
#This line has already been run once to generate the 2 files, which I have also included in the Repo 
#data_split('datasets/sodabottles/train.csv')


def mode_classes(mode):
    if mode == "full":
        return ['P.Orig', 'MD.Orig', 'P.Cherry', 'P.Zero']
    elif mode == "no_pepsi":
        return ['MD.Orig', 'P.Cherry', 'P.Zero']
    elif mode == "no_cherry":
        return ['P.Orig', 'MD.Orig', 'P.Zero']
    else:
        raise Exception("Unknown mode used")


class bottle_test(torch.utils.data.Dataset):
    def __init__(self, filename, transforms = None, path='sodabottle/', mode="full"):
        self.root_dir = os.path.join(os.getcwd(), 'datasets', path)
        self.frames = pd.read_csv(os.path.join(self.root_dir, 'labels', filename))
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,'images',self.frames.iloc[index,1])
        label = self.frames.iloc[index,0]
        ## according to alphabetic
        labels = mode_classes(self.mode)

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, labels.index(label)

    def __len__(self):
        return len(self.frames)


class bottle(torch.utils.data.Dataset):
    def __init__(self, filename, transforms = None, path='sodabottles/', mode="full"):
        self.root_dir = os.path.join(os.getcwd(), 'datasets', path)
        self.frames = pd.read_csv(filename)
        self.transforms = transforms
        self.mode = mode

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,'train',self.frames.iloc[index,1])
        label = self.frames.iloc[index,0]
        ## according to alphabetic
        labels = mode_classes(self.mode)

        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img, labels.index(label)

    def __len__(self):
        return len(self.frames)
