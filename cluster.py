# cluster
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
from wikiart import WikiArtDataset, WikiArtPart2
import torcheval.metrics as metrics
import json
import argparse
import numpy as np
import torchvision.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file",
                    default="config.json")

args = parser.parse_args()

config = json.load(open(args.config))

testingdir = config["testingdir"]
device = config["device"]

print("Running...")

def get_encodings(modelfile=None, device="cpu"):

    loader = DataLoader(testingdataset, batch_size=1)
    model = WikiArtPart2()
    model.load_state_dict(torch.load(modelfile, weights_only=True))
    model = model.to(device)
    model.eval()

    encodings = list()
    for batch in tqdm.tqdm(loader):
        X, _ = batch
        encoded_img, _ = model(X, return_encoded=True)
        encodings.extend(encoded_img.numpy(force=True))

    return encodings

def cluster(encodings):
    transform = T.ToPILImage()
    #print(type(encodings))
    from matplotlib import pyplot as plt
    plt.imshow(np.transpose(encodings[0], (1,2,0)), interpolation='nearest')
    plt.show()
    #print(encodings[0].shape, type(encodings[0]))
    #img = transform(encodings[0])
    #img.show()

if os.path.isfile(config["encodingsfile"]):
    print('Loading encodings from file...')
    encodings = np.load(config["encodingsfile"])
else:
    print('Creating encodings using model')
    testingdataset = WikiArtDataset(testingdir, device)
    encodings = get_encodings(modelfile=config["modelfile2"], device=device)
    if config['encodingsfile']:
        np.save(config['encodingsfile'], encodings)

cluster(encodings)

