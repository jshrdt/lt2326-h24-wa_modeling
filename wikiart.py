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
                                                 self.filename)).float()
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
        self.classes = sorted(list(classes))
        self.device = device
        self.label_counts = label_counts
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
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
