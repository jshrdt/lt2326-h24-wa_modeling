# generate

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtPart3
import json
import argparse
import numpy as np
import os

#https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file",
                    default="config.json")

args, unknown = parser.parse_known_args()
config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config['device'] if torch.cuda.is_available() else 'cpu'

print("Running...")

traindataset = WikiArtDataset(trainingdir, device)

# def gram_matrix(input):
#     a, b, c, d = input.size()  # a=batch size(=1)
#     # b=number of feature maps
#     # (c,d)=dimensions of a f. map (N=c*d)

#     features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

#     G = torch.mm(features, features.t())  # compute the gram product

#     # we 'normalize' the values of the gram matrix
#     # by dividing by the number of element in each feature maps.
#     res = G.div(a * b * c * d).flatten()

#     return res

def train_embeds(loader, epochs=5, device="cpu"):
    print('\nTraining art style embeddings...')
    model = WikiArtPart3(traindataset.style2idx, device, num_classes=27).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            _, output = model(y.to(device), train_embeds=True) #y to select style embedding (out of n=27 classes)
            style_loss = criterion(output, X)
            style_loss.backward()
            accumulate_loss += style_loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    return model


def train(loader, model, epochs=5, modelfile=None, device="cpu"):
    print('\nTraining style transfer...')
    # freeze embeddings layer & remove it from optimizer.
    model.style_embeds.weight.requires_grad = False
    optimizer = Adam([param for param in model.parameters()
                      if param.requires_grad == True], lr=0.01)
    # Initiliase loss for content image and artstyle.
    criterion_img = nn.MSELoss()
    criterion_style = nn.MSELoss()

    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            encoded, decoded, style_embed = model(y, content_imgs=X)

            # content loss
            c_loss = criterion_img(decoded, X)
            #style loss
            s_loss = criterion_style(style_embed, X)
            loss = c_loss + s_loss
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)
        print('Model saved to file.')

    return model

def combine(model, img_idx, transfer_style):
    # run in interactive window to view example img &
    # image post style transfer
    model.eval()
    img = traindataset[img_idx][0]
    altered_img = model(transfer_style, content_imgs=img)[1]

    # Plot original & altered image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.imshow(img.numpy(force=True).transpose(1, 2, 0))
    ax2.imshow(altered_img[0].numpy(force=True).transpose(1, 2, 0))
    plt.show()
    print('--- end ---')


if __name__=='__main__':
    class_weights = torch.Tensor([1 / traindataset.label_counts[label]
                                    for label in traindataset.labels_str])
    sampler = WeightedRandomSampler(weights=class_weights,
                                    num_samples=len(traindataset))
    loader = DataLoader(traindataset, batch_size=config["batch_size"],
                        sampler=sampler)
    # Get model
    if os.path.isfile(config["modelfile3"]):
        print('Loading model from file')
        # load model with embeds layer from file
        model = WikiArtPart3(traindataset.style2idx, device)
        model.load_state_dict(torch.load(config["modelfile3"], weights_only=True))
        model = model.to(device)
        model.eval()
    else:
        model_with_embeds = train_embeds(loader, epochs=config['epochs'], device=device)
        model = train(loader, model_with_embeds, epochs=config['epochs'], modelfile=config["modelfile3"],
                       device=device)

    print('Done')

    # Run in interactive window to view original and altered image.
    combine(model, img_idx=4, transfer_style='Ukiyo_e')
