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
            # added rescaling to range 0-1
            self.image = read_image(os.path.join(self.imgdir, self.label,
                                                 self.filename)).float()/255
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
        # Fix label encoding across train/testsets
        self.label_to_idx = {label: i for i, label in enumerate(self.classes)}
        self.labels_str = labels
        self.labels = [self.label_to_idx[label] for label in labels]
        # for part 3
        self.style2idx = {arttype: i for i, arttype in enumerate(self.classes)}

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


### Part 2 ###
class WikiArtPart2(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(9, 3, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            )
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(3, 9, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Sigmoid()
            )

    def forward(self, x, decode_only=False):
        if decode_only:
            decoded = self.decoder(x)
            return decoded
        else:
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return encoded, decoded


### Part 3 ###
class WikiArtPart3(nn.Module):
    def __init__(self, style2idx, device='cpu', num_classes=27):
        super().__init__()
        self.style2idx = style2idx
        self.device = device
        self.style_embeds = nn.Embedding(num_classes, 416*416*3)

        self.style_encoder = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(9, 3, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            )
        self.style_decoder = nn.Sequential( 
            nn.ConvTranspose2d(3, 9, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Sigmoid()
            )

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 9, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(9, 3, kernel_size=5, padding=1),
            nn.MaxPool2d(2, stride=2, padding=1),
            )
        self.decoder = nn.Sequential( 
            nn.ConvTranspose2d(3, 9, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReLU(),
            nn.ConvTranspose2d(9, 3, kernel_size=5, padding=2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Sigmoid()
            )

    def forward(self, style_idx, content_imgs=None, train_embeds=False):
        # Embeddings training
        if train_embeds:
            # Retrieve embeddings using artstyle as idx, reshape to WikiArt
            # image dimensions (batch_size, 23, 416, 416)
            style_embeds = self.style_embeds(style_idx).reshape(
                style_idx.size()[0], 3, 416, 416)

            # Pass through network 
            style_encode = self.style_encoder(style_embeds)
            style_decode = self.style_decoder(style_encode)

            return style_encode, style_decode

        # Style transfer task
        else:
            # If style_idx is string of artstyle: retrieve singe artstyle embed
            # for style transfer to single content image.
            if type(style_idx)==str:
                style_idx = self.style2idx[style_idx]
                embeds = self.style_embeds(
                    torch.tensor(style_idx).to(self.device)
                    ).reshape(1, 3, 416, 416)
            # Style_idx is of batch_size (y tensors from dataloader batches),
            # retrieve batch_size of varying style embeddings.
            else:
                embeds = self.style_embeds(
                    style_idx.to(self.device)
                    ).reshape(style_idx.size()[0], 3, 416, 416)
            # Single content image passed (non-batched), reshape to fit expected
            # network input dimensions.
            if len(content_imgs.size())==3:
                content_imgs = content_imgs.reshape(1, 3, 416, 416)

            # Concat content image(s) and style embedding(s) along channel dim.
            input = torch.cat((content_imgs, embeds), 1)
            # Pass input through network
            encoded = self.encoder(input)
            decoded = self.decoder(encoded)

            return encoded, decoded, embeds