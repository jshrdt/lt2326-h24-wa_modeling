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
from wikiart import WikiArtDataset, WikiArtModel
import json
import argparse

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
#testingdataset = WikiArtDataset(testingdir, device)


#the_image, the_label = traindataset[5]
#print(the_image, the_image.size())

# the_showable_image = F.to_pil_image(the_image)
# print("Label of img 5 is {}".format(the_label))
# the_showable_image.show()

def train(epochs=3, batch_size=32, modelfile=None, device="cpu",
          mode=None):
    loader = DataLoader(traindataset, batch_size=batch_size, shuffle=True)
    model = WikiArtModel(mode=mode).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)

    class_weights1 = [(len(traindataset.filedict)
                       /traindataset.label_counts[label]
                       / len(traindataset.classes))
                      for label in traindataset.classes]
    class_weights2 = [(len(traindataset.filedict)
                       /traindataset.label_counts[label])
                      for label in traindataset.classes]
    class_weights3 = [1 / traindataset.label_counts[label]
                      for label in traindataset.classes]
    criterion = nn.NLLLoss(weight=torch.Tensor(class_weights3)).to(device)
    # weight = torch.Tensor(class_weights1)

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            X, y = batch
            y = y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)

    return model

model = train(config["epochs"], config["batch_size"],
              modelfile=config["modelfile"], device=device,
              mode=config["mode"])
