# generate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtPart3
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
args, unknown = parser.parse_known_args()
config = json.load(open(args.config))

device = config['device'] if torch.cuda.is_available() else 'cpu'
traindataset = WikiArtDataset(config['trainingdir'], device)
testingdataset = WikiArtDataset(config['testingdir'], device)


def train_embeds(loader: DataLoader, epochs: int = 5,
                 device: str = 'cpu') -> WikiArtPart3:
    """Initialise model and train art style embeddings, return model."""
    print('\nTraining art style embeddings...')
    model = WikiArtPart3(traindataset.style2idx, device,
                         num_classes=27).to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print('Starting epoch {}'.format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            # use y to index style embedding in embedding matrix (27)
            _, output = model(y.to(device), train_embeds=True)
            style_loss = criterion(output, X)
            style_loss.backward()
            accumulate_loss += style_loss
            optimizer.step()

        print('In epoch {}, loss = {}'.format(epoch, accumulate_loss))

    return model


def train(loader: DataLoader, model: WikiArtPart3, epochs: int = 5,
          modelfile: str = None) -> WikiArtPart3:
    """Train the transfer network of the model, (save), and return model."""
    print('\nTraining style transfer...')
    # Freeze embeddings layer & remove it from optimizer.
    model.style_embeds.weight.requires_grad = False
    optimizer = Adam([param for param in model.parameters()
                      if param.requires_grad == True], lr=0.01)
    # Initiliase loss for content image and artstyle.
    criterion_img = nn.MSELoss()
    criterion_style = nn.MSELoss()

    for epoch in range(epochs):
        print('Starting epoch {}'.format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, y = batch
            # Pass content images and true y to model, uses y as to index
            # matching art style embedding from matrix.
            _, decoded, style_embed = model(y, content_imgs=X)
            # Content loss based on original input images
            c_loss = criterion_img(decoded, X)
            # Style loss based on style embedding
            s_loss = criterion_style(style_embed, X)
            # Combine losses to optimise for both content & style, update
            loss = c_loss + s_loss
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print('In epoch {}, loss = {}'.format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)
        print('Model saved to file.')

    return model


def combine(model: WikiArtPart3, img_idx: int, transfer_style: str) -> None:
    """Transfers style as indicated by art genre to content image indexed
    from testdata."""
    # Freeze model, retrieve original image, and transfer style to image.
    model.eval()
    img = testingdataset[img_idx][0]
    altered_img = model(transfer_style, content_imgs=img)[1]

    # Plot original & altered image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    ax1.imshow(img.numpy(force=True).transpose(1, 2, 0))
    ax2.imshow(altered_img[0].numpy(force=True).transpose(1, 2, 0))
    plt.show()


if __name__=='__main__':
    print('Running...')
    # Load data, fix class imbalances with class weights.
    class_weights = torch.Tensor([1 / traindataset.label_counts[label]
                                  for label in traindataset.labels_str])
    sampler = WeightedRandomSampler(weights=class_weights,
                                    num_samples=len(traindataset))
    loader = DataLoader(traindataset, batch_size=config['batch_size'],
                        sampler=sampler)

    # Get model from file or train new.
    if os.path.isfile(config['modelfile3']):
        print('Loading model from file')
        model = WikiArtPart3(traindataset.style2idx, device)
        model.load_state_dict(torch.load(config['modelfile3'],
                                         weights_only=True))
        model = model.to(device)
    else:
        embed_model = train_embeds(loader, epochs=config['epochs'],device=device)
        model = train(loader, embed_model, epochs=config['epochs'],
                      modelfile=config['modelfile3'], device=device)

    # Run in interactive window to view original and altered image.
    combine(model, img_idx=4, transfer_style='High_Renaissance')
    print('--- end ---')

