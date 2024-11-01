# generate

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtPart2, WikiArtPart3
import json
import argparse
import numpy as np

#https://pytorch.org/tutorials/advanced/neural_style_tutorial.html


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file",
                    default="config.json")

args, unknown = parser.parse_known_args()
config = json.load(open(args.config))

trainingdir = config["trainingdir"]
testingdir = config["testingdir"]
device = config["device"]

print("Running...")

traindataset = WikiArtDataset(trainingdir, device, part3=False)
class StyleEmbedder(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()
        self.style_embeds = nn.Embedding(num_classes, 416*416*3)
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
    def forward(self, style_idx):
        style_embed = self.style_embeds(style_idx).reshape(
            style_idx.size()[0], 3, 416, 416)
        style_encode = self.encoder(style_embed)
        style_decode = self.decoder(style_encode)

        return style_encode, style_decode

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

def train_embeds(loader, epochs=5, modelfile=None, device="cpu"):
    print('get style embeds')
    # model = WikiArtPart3(len(traindataset)).to(device)
    model = StyleEmbedder(num_classes=27).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()# nn.functional.mse_loss()
    #criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        encodings=list()
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            
            encoded, output = model(y.to(device)) #y to select style embedding (out of n=27 classes)
            
            style_loss = criterion(output, X)
            style_loss.backward()
            accumulate_loss += style_loss
            optimizer.step()
            # if batch_id>1:
                # print(innbetween == embeds.weight)
        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    return model.style_embeds
            

class WikiArtPart3(nn.Module):
    def __init__(self, style_embeddings, num_classes=27):
        super().__init__()
        self.embeddings = style_embeddings
        #print(type(style_embeddings))
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

    def forward(self, x, arttype, decode_only=False):
        if decode_only:
            decoded = self.decoder(x)
            return decoded
        else:
            if type(arttype)==str:
                arttype = traindataset.style2idx[arttype]
               # print(arttype, type(arttype))
                style_embed = self.embeddings(torch.tensor(arttype).to(device)).reshape(1, 3, 416, 416)
            else:
                style_embed = self.embeddings(arttype.to(device)).reshape(arttype.size()[0], 3, 416, 416)
            if len(x.size())==3:
                x = x.reshape(1, 3, 416, 416)

            input = torch.cat((x, style_embed), 1)
            #print(input.size())
            encoded = self.encoder(input)
            decoded = self.decoder(encoded)
            return encoded, decoded, style_embed
        
def train(loader, embeddings, epochs=5, modelfile=None, device="cpu"):

    model = WikiArtPart3(embeddings, len(traindataset)).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion_img = nn.MSELoss()
    criterion_style = nn.MSELoss()
    #criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        encodings=list()
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            encoded, decoded, style_embed = model(X, y)

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

    return model

def combine(img, style):
    model.eval()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.imshow(img.numpy(force=True).transpose(1, 2, 0))
    ax2.imshow(model(img, style)[1][0].numpy(force=True).transpose(1, 2, 0))
    plt.show()


if __name__=='__main__':
    class_weights = torch.Tensor([1 / traindataset.label_counts[label]
                                    for label in traindataset.labels_str])
    sampler = WeightedRandomSampler(weights=class_weights,
                                    num_samples=len(traindataset))
    loader = DataLoader(traindataset, batch_size=config["batch_size"],
                        sampler=sampler)

    embeds = train_embeds(loader, epochs=5, device=device) #save embeds?

    model, X, encoed, decoded = train(loader, embeds, epochs=5,
                modelfile=config["modelfile3"],
                device=device)

    #combine(X[0], 'Ukiyo_e')
