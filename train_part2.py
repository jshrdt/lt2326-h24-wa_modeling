import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
import tqdm
from wikiart import WikiArtDataset, WikiArtPart2
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file",
                    default="config.json")

args = parser.parse_args()
config = json.load(open(args.config))
trainingdir = config["trainingdir"]
device = config["device"]

print("Running...")
traindataset = WikiArtDataset(trainingdir, device)

def train(epochs=3, batch_size=32, modelfile=None, device="cpu"):
    """Train an autoencoder by minimising MSE loss between decoded image
    and original image, return (and save) model."""
    # Get class weights in inverse to class frequency and load accordingly.
    class_weights = torch.Tensor([1 / traindataset.label_counts[label]
                                  for label in traindataset.labels_str])
    sampler = WeightedRandomSampler(weights=class_weights,
                                    num_samples=len(traindataset))
    loader = DataLoader(traindataset, batch_size=batch_size, sampler=sampler)

    # Initiliase model, optimiser, loss function.
    model = WikiArtPart2().to(device)
    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Start training.
    for epoch in range(epochs):
        print("Starting epoch {}".format(epoch))
        accumulate_loss = 0
        for batch_id, batch in enumerate(tqdm.tqdm(loader)):
            optimizer.zero_grad()
            X, _ = batch
            _, output = model(X)
            # Minimise MSE loss between decoded image and original.
            loss = criterion(output, X)
            loss.backward()
            accumulate_loss += loss
            optimizer.step()

        print("In epoch {}, loss = {}".format(epoch, accumulate_loss))

    if modelfile:
        torch.save(model.state_dict(), modelfile)
        print('Model saved to file.')

    return model

if __name__=='__main__':
    # Train & save autoencoder
    model = train(config["epochs"], config["batch_size"],
                  modelfile=config["modelfile2"], device=device)
    print('--- end ---')