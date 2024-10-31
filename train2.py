import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtPart2
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file",
                    default="config.json")

args = parser.parse_args()
config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")

traindataset = WikiArtDataset(trainingdir, device)


def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    class_weights = torch.Tensor([1 / traindataset.label_counts[label]
                                  for label in traindataset.labels_str])

    sampler = WeightedRandomSampler(weights=class_weights,
                                    num_samples=len(traindataset))

    loader = DataLoader(traindataset, batch_size=batch_size, sampler=sampler)
    model = WikiArtPart2().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        encodings=list()
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, _ = batch
            optimizer.zero_grad()
            _, output = model(X)
            loss = criterion(output, X)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(config["epochs"], config["batch_size"],
              modelfile=config["modelfile2"], device=device)