# cluster
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import tqdm
from wikiart import WikiArtDataset, WikiArtPart2
import json
import argparse
import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', help='configuration file',
                    default='config.json')
args, unknown = parser.parse_known_args()

config = json.load(open(args.config))
device = config['device']

print('Running...')

def get_encodings(testingdir: str = config['testingdir'], device='cpu',
                  predict: bool = True) -> dict:
    """Loads data from testdir and encodes images with model. Returns a 
    dictionary with encoded images and their gold class labels."""
    # Load test data
    testingdataset = WikiArtDataset(testingdir, device)
    loader = DataLoader(testingdataset, batch_size=1)
    # Initialise encodings dict
    encodings_dict = {'encodings': list(), 'y_gold': list()}
    # Get encodings and gold labels
    for batch in tqdm.tqdm(loader):
        X, true_y = batch
        encodings_dict['y_gold'].append(true_y)
        # Skip encoding step if encodings were pre-loaded, run function
        # only to retrieve gold labels.
        if predict:
            encoded_img, _ = model(X)
            encodings_dict['encodings'].append(encoded_img)

    return encodings_dict

def decode(encoding: np.ndarray, model: WikiArtPart2) -> np.ndarray:
    """Helper function to decode an encoded image from encodings dict,
    output suited for plt.imshow()."""
    decoded_img = model(torch.tensor(encoding).to(device),
                    decode_only=True)[0].numpy(force=True).transpose(1, 2, 0)
    return decoded_img

def format_encodings(encodings: pd.Series) -> list:
    """Flatten and scale encodings from pd.Series, return as list."""
    encodings_flat = encodings.apply(np.ndarray.flatten)
    scaler = StandardScaler()
    scaler.fit(np.array(list(encodings_flat)))
    encodings_scaled = [scaler.transform(encoding.reshape(1, -1))[0]
                        for encoding in encodings_flat]

    return encodings_scaled

def cluster(encodings_scaled: list) -> np.ndarray:
    """Run KMeans on a PCA values of from flattened and scaled image
    encodings, return array of KMeans cluster labels."""
    # collapse flat/scaled image encodings to singular value.
    pca = PCA()
    pca.fit(encodings_scaled)
    # Fit KMeans on pca 
    kmeans = KMeans(n_clusters=27)
    kmeans.fit(pca.singular_values_.reshape(-1, 1))
    
    
    clust_labels = kmeans.predict(np.float64(pca.singular_values_).reshape(-1, 1))

    return clust_labels, pca.singular_values_

def plot_cluster(data_df, limit=False):
    # limit df due to outliers
    if limit:
        data_df = data_df[data_df['encodings_PCA']<limit]
    # plot kmeans class (y-ax) to image's PCA value (x-ax), colour-coded
    # by original gold label.
    fig = plt.figure()
    fig.set_figwidth(15)
    ax = fig.add_subplot(111)
    scatter = ax.scatter(list(data_df['encodings_PCA']), data_df['y_pred'],
                        c=data_df['y_gold'],
                        cmap=plt.cm.hsv)
    # Add information
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('Image PCA value')
    ax.set_ylabel('KMeans cluster ID')

    plt.colorbar(scatter, label='True class from dataset')


# load model
model = WikiArtPart2()
model.load_state_dict(torch.load(config['modelfile2'], weights_only=True))
model = model.to(config['device'])
model.eval()

# Get encodings and their true_y, either loaded from file, or created anew.
if  os.path.isfile(config['encodingsfile']):
    print('Loading encodings from file...')
    encodings = np.load(config['encodingsfile'])
    encodings_dict = get_encodings(predict=False)
    encodings_dict['encodings'] = list(encodings)
else:
    print('Encoding images from testdir using model...')
    encodings_dict = get_encodings(device=config['device'])
    encodings_matrix = np.array([encodings_dict['encodings'][i].numpy(force=True)
                                for i in range(len(encodings_dict['encodings']))])
    if config['encodingsfile']:
        np.save(config['encodingsfile'], encodings_matrix)

##plt.imshow(decode(encodings_dict['encodings'][6], model))

# cluster and plot encoded images and respective 
data_df = pd.DataFrame(encodings_dict)

# Flatten and scale encodings
scaled_encodings = format_encodings(data_df['encodings'])
data_df.insert((data_df.shape[1]),'encodings_scaled', scaled_encodings)

# Get pca values from fitting on scaled encodings
clusters, pcas = cluster(scaled_encodings)
data_df.insert((data_df.shape[1]),'y_pred', pd.DataFrame(clusters))
data_df.insert((data_df.shape[1]), 'encodings_PCA', pcas)

# Plot clusterXpcaXgold, adjust limit for outliers (max outlier at 2724, 
# more values from 1200, most are below 450/200)
#plot_cluster(data_df, limit=200)
