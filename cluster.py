# cluster
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import tqdm
from wikiart import WikiArtDataset, WikiArtPart2
import json
import argparse
import numpy as np
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


def format_encodings(encodings: pd.Series) -> list:
    """Flatten and standard scale encodings, return as list of ndarrays."""
    print('Formatting encodings...')
    encodings_flat = encodings.apply(np.ndarray.flatten)
    scaler = StandardScaler()
    scaler.fit(np.array(list(encodings_flat)))
    encodings_scaled = [scaler.transform(encoding.reshape(1, -1))[0]
                        for encoding in encodings_flat]

    return encodings_scaled


def pca_reduce(encodings_scaled: list) -> np.ndarray:
    """Apply principal component analysis to scaled encodings."""
    print('Running PCA...')
    # Reduce each encoding to a single value.
    pca = PCA(n_components=1)
    pca_encodings = pca.fit_transform(encodings_scaled)

    return pca_encodings


def cluster(encodings_scaled: list) -> np.ndarray:
    """Run KMeans on flattened and scaled image encodings, return clusters."""
    # Fit and predict on formatted encodings.
    print('Running KMeans...')
    kmeans = KMeans(n_clusters=27)
    kmeans.fit(encodings_scaled)
    clust_labels = kmeans.predict(np.float64(encodings_scaled))

    return clust_labels


def plot_cluster(data_df):
    """Plot PCA-reduced image encodings against predicted clusters, colour-
    (and shape to an extent) coded by true art style class."""
    # Create wide rectangle plot to spread data points apart
    fig = plt.figure()
    fig.set_figwidth(15)
    ax = fig.add_subplot(111)

    # Set marker style variation & color for gold class
    m = ['^', 'o', '*'] * int(27/3)
    cmap = plt.colormaps['hsv']
    color_list = cmap(np.linspace(0, 1, 27))

    # Plot image's PCA value (x-ax) against its kmeans class (y-ax).
    for i in range(len(data_df)):
        plt.scatter(data_df['encodings_PCA'][i],
                    data_df['cluster_pred'][i],
                    marker=m[int(data_df['y_gold'][i])],
                    color=color_list[int(data_df['y_gold'][i])]
                    ).set_cmap('hsv')

    # Add information to fig
    ax.set_title('K-Means clustering on scaled encodings')
    ax.set_xlabel('Image PCA value')
    ax.set_ylabel('KMeans cluster ID')
    cbar = plt.colorbar(label='True class from dataset')
    cbar.set_ticks(np.arange(0, 1, (1/7)))
    cbar.set_ticklabels([str(idx) for idx in range(0, 27)[::4]])

    plt.show()


if __name__=='__main__':
    # Load model
    model = WikiArtPart2()
    model.load_state_dict(torch.load(config['modelfile2'], weights_only=True))
    model = model.to(config['device'])
    model.eval()

    # Get encodings and their true class, either loaded from file, or created anew.
    if  os.path.isfile(config['encodingsfile']):
        print('Loading encodings from file...')
        encodings = np.load(config['encodingsfile'])
        encodings_dict = get_encodings(predict=False)
        encodings_dict['encodings'] = list(encodings)
    else:
        # Use model to encode images in testdir
        print('Encoding images from testdir using model...')
        encodings_dict = get_encodings(device=config['device'])
        encodings_matrix = np.array(
            [encodings_dict['encodings'][i].numpy(force=True)
             for i in range(len(encodings_dict['encodings']))])
        # Save encodings to file
        if config['encodingsfile']:
            np.save(config['encodingsfile'], encodings_matrix)

    ## Cluster and plot encoded images
    # Format data in df
    data_df = pd.DataFrame(encodings_dict)

    # Flatten and scale encodings
    scaled_encodings = format_encodings(data_df['encodings']) # 630, 32448
    data_df['encodings_scaled'] = scaled_encodings

    # cluster encodigs, then perform PCA for plotting against clusters
    data_df['cluster_pred'] = cluster(scaled_encodings)
    data_df['encodings_PCA'] = pca_reduce(scaled_encodings)

    # plot clusters against pca reduced encodings
    plot_cluster(data_df)
    print('--- end ---')
