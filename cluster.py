# cluster
import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
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

def cluster(encodings_scaled: list, cluster_pca=False) -> np.ndarray:
    """Run KMeans on flattened and scaled image encodings or on their
    PCA values, return array of KMeans cluster labels (and pca values,
    if cluster_pca==True)."""

    if cluster_pca==True:
        print('Runnning PCA...')
        # Collapse flat/scaled image encodings to singular value.
        pca = PCA()
        pca.fit(encodings_scaled)

        # Fit & predcit KMeans on pca reduced images
        print('Running KMeans...')
        kmeans = KMeans(n_clusters=27)
        kmeans.fit(pca.singular_values_.reshape(-1, 1))
        clust_labels = kmeans.predict(np.float64(pca.singular_values_).reshape(-1, 1))
        return clust_labels, pca.singular_values_

    else:
        # Fit and predict on formatted encodings.
        print('Running KMeans...')
        kmeans = KMeans(n_clusters=27)
        kmeans.fit(encodings_scaled)
        clust_labels = kmeans.predict(np.float64(encodings_scaled))

        return clust_labels

def plot_cluster(data_df, plt_title='K-Means clustering', limit=False):
    # limit df due to outliers
    if limit:
        data_df = data_df[data_df['encodings_PCA']<limit].reset_index()
    # plot kmeans class (y-ax) to image's PCA value (x-ax), colour-coded
    # by original gold label.
    fig = plt.figure()
    fig.set_figwidth(15)
    ax = fig.add_subplot(111)
    # Set marker style variation & color
    m = ['^', 'o', '*'] * int(27/3)
    cmap = plt.colormaps['hsv']
    color_list = cmap(np.linspace(0, 1, 27))
    # plot
    for i in range(len(data_df)):
        plt.scatter(data_df['encodings_PCA'][i], data_df['y_pred'][i],
                    marker=m[int(data_df['y_gold'][i])], color=color_list[int(data_df['y_gold'][i])]
                    )

    # Add information
    ax.set_title(plt_title)
    ax.set_xlabel('Image PCA value')
    ax.set_ylabel('KMeans cluster ID')
    cbar = plt.colorbar(label='True class from dataset')
    cbar.set_ticks(np.arange(0, 1, (1/7)))
    cbar.set_ticklabels([str(idx) for idx in range(0, 27)[::4]])

    plt.show()


if __name__=='__main__':
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

    # cluster and plot encoded images and respective 
    data_df = pd.DataFrame(encodings_dict)

    # Flatten and scale encodings
    print('Formatting encodings...')
    scaled_encodings = format_encodings(data_df['encodings'])
    data_df.insert((data_df.shape[1]),'encodings_scaled', scaled_encodings)

    # Set whether to cluster formatted encodings or their PCA values
    cluster_pca=True

    if cluster_pca:
        # cluster PCA values and plot against clusters
        clusters, pcas = cluster(scaled_encodings, cluster_pca=True)
        data_df.insert((data_df.shape[1]),'y_pred', pd.DataFrame(clusters))
        data_df.insert((data_df.shape[1]), 'encodings_PCA', pcas)
        title = 'K-Means clustering on image PCA values'
    else:
        # cluster encodigs, then perform PCA for plotting against clusters
        clusters = cluster(scaled_encodings)
        data_df.insert((data_df.shape[1]),'y_pred', pd.DataFrame(clusters))
        print('Running PCA...')
        pca = PCA()
        pca.fit(scaled_encodings)
        data_df['encodings_PCA'] = pca.singular_values_
        title = 'K-Means clustering on scaled encodings'


    plot_cluster(data_df, plt_title=title, limit=False)
    plot_cluster(data_df, plt_title=title, limit=400)
    print('--- end ---')
