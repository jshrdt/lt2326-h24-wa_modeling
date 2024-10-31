import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm

import random

class WikiArtImage:
    def __init__(self, imgdir, label, filename):
        self.imgdir = imgdir
        self.label = label
        self.filename = filename
        self.image = None
        self.loaded = False

    def get(self):
        if not self.loaded:
            self.image = read_image(os.path.join(self.imgdir, self.label,
                                                 self.filename)).float()/255
            #print(os.path.join(self.imgdir, self.label,
             #                                    self.filename))
            self.loaded = True

        return self.image

class WikiArtDataset(Dataset):
    def __init__(self, imgdir, device="cpu"):
        walking = os.walk(imgdir)
        filedict = {}
        indices = []
        classes = set()
        label_counts = dict()
        labels = list()
        print("Gathering files for {}".format(imgdir))
        for item in walking:
            sys.stdout.write('.')
            arttype = os.path.basename(item[0])
            artfiles = item[2]
            for art in artfiles:
                filedict[art] = WikiArtImage(imgdir, arttype, art)
                label_counts[arttype] = label_counts.get(arttype, 0) +1
                indices.append(art)
                classes.add(arttype)
                labels.append(arttype)
        print("...finished")
        self.filedict = filedict
        self.imgdir = imgdir
        self.indices = indices
       # random.shuffle(indices)
        self.classes = sorted(list(classes))
        self.device = device
        self.label_counts = label_counts
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels_str = labels
        self.labels = [self.label_to_idx[label] for label in labels]

    def __len__(self):
        return len(self.filedict)

    def __getitem__(self, idx):
        imgname = self.indices[idx]
        imgobj = self.filedict[imgname]
        ilabel = self.label_to_idx[imgobj.label]
        image = imgobj.get().to(self.device)

        return image, ilabel

class WikiArtModel(nn.Module):
    def __init__(self, mode=None, num_classes=27):
        super().__init__()
        self.mode = mode
        if self.mode == 'bonusA':
            self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
            self.pool = nn.AdaptiveAvgPool2d((50,50))
            self.flatten = nn.Flatten()
            self.batchnorm1d = nn.BatchNorm1d(50*50)
            self.linear1 = nn.Linear(50*50, 300)
            self.dropout = nn.Dropout(0.01)
            self.activfunc = nn.Sigmoid()
            self.linear2 = nn.Linear(300, num_classes)  # 27
            self.softmax = nn.LogSoftmax(dim=1)

        else:
            self.conv2d = nn.Conv2d(3, 1, (4,4), padding=2)
            self.pool = nn.MaxPool2d((4,4), padding=2)
            self.flatten = nn.Flatten()
            self.batchnorm1d = nn.BatchNorm1d(105*105)
            self.linear1 = nn.Linear(105*105, 300)
            self.dropout = nn.Dropout(0.01)
            self.activfunc = nn.ReLU()
            self.linear2 = nn.Linear(300, num_classes)  # 27
            self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, image):
        output = self.conv2d(image)
        #print("convout {}".format(output.size()))
        output = self.pool(output)
        #print("poolout {}".format(output.size()))        
        output = self.flatten(output)
        output = self.batchnorm1d(output)
        #print("poolout {}".format(output.size()))        
        output = self.linear1(output)
        output = self.dropout(output)
        output = self.activfunc(output)
        output = self.linear2(output)
        return self.softmax(output)
    
class WikiArtPart2(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        
    # works & produces 3x3 imgs  ; https://github.com/E008001/Autoencoder-in-Pytorch
        # self.encoder = torch.nn.Sequential(
        #    torch.nn.Conv2d(3, 16, 4, stride=1, padding=2),  # 
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, stride=1),
        #     torch.nn.Conv2d(16, 1, 4, stride=1, padding=2),  # b, 8, 3, 3
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )

        # self.decoder = torch.nn.Sequential(
        #     torch.nn.Upsample(scale_factor=1, mode='nearest'),
        #     torch.nn.Conv2d(1, 16, 4, stride=1,  padding=1),  # b, 16, 10, 10
        #     torch.nn.ReLU(),
        #     torch.nn.Upsample(scale_factor=1, mode='nearest'),
        #     torch.nn.Conv2d(16, 3, 4, stride=1,  padding=2),  # b, 8, 3, 3
        #     torch.nn.Sigmoid()
        # )
# my og
        self.encoder = nn.Sequential(
            nn.Conv2d(3,9, kernel_size=5, padding=1),
            nn.AvgPool2d(2, stride=2, padding=1),
            nn.ReLU(),
             
            nn.Conv2d(9,3, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            
            )
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(3,9, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            
            nn.ConvTranspose2d(9,3, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Sigmoid()
            )
         #   self.flatten = nn.Flatten()
       # self.linear1 = nn.Linear(519168, num_classes)
       # self.sigmoid = nn.Sigmoid()


    def forward(self, x, return_encoded=False):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        if return_encoded:
            return encoded, decoded
        else:
            return decoded
    
