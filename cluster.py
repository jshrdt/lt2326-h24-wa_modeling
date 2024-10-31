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

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

    encodings_dict = dict()
    for i, batch in tqdm.tqdm(enumerate(loader)):
        X, true_y = batch
        encoded_img, pred_y = model(X, return_encoded=True)
        encodings_dict[i] = {'img': encoded_img.numpy(force=True), 'label': true_y}

    return encodings_dict

def cluster(encodings_dict):
    # color for each encoding by its y label -> visualy check if genres cluster

    #print(max(encodings[0].flatten()), min(encodings[0].flatten()))
    #transform = T.ToPILImage()
    #print(type(encodings))
    #plt.imshow(np.transpose(encodings[0], (1,2,0)), interpolation='nearest')
    #plt.show()
    #print(encodings[0].shape, type(encodings[0]))
    #img = transform(encodings[0])
    #img.show()
    encodings = [img_dict['img'].flatten() for img_dict in encodings_dict.values()]
    #print(encodings[0].shape)
    
    true_labels = [img_dict['label'] for img_dict in encodings_dict.values()]

    kmeans = KMeans(n_clusters=27, init='k-means++', random_state=0)
    labels = kmeans.fit_predict(encodings) #2mins for n=5, 50sec for n=2
    
    # xs = encodings[:,0]
    # ys = encodings[:,1]
    
    #plt.scatter(c=labels,alpha=0.5)
    # Assign the cluster centers: centroids
    # centroids = kmeans.cluster_centers_
    # # Assign the columns of centroids: centroids_x, centroids_y
    # centroids_x = centroids[:,0]
    # centroids_y = centroids[:,1]
    # Make a scatter plot of centroids_x and centroids_y
   # plt.scatter(marker='D',s=50)
   # plt.show()


    clusters_dict = {cluster_id : list() for cluster_id in set(labels)}
    #for i, cluster_id in enumerate(labels):
     #   print('true:', true_labels[i], 'pred:', cluster_id)
        #clusters_dict[cluster_id].append(encodings[i])

    return labels, clusters_dict

# if os.path.isfile(config["encodingsfile"]):
#     print('Loading encodings from file...')
#     encodings = np.load(config["encodingsfile"])
# else:
print('Creating encodings using model')
testingdataset = WikiArtDataset(testingdir, device)
encodings_dict = get_encodings(modelfile=config["modelfile2"], device=device)
#print(type(encodings), type(encodings[0]), type(encodings[0][0]))


# if config['encodingsfile']:
#     np.save(config['encodingsfile'], encodings) ##

Y, clusters = cluster(encodings_dict)
#print([('cluster nr: '+ str(id), 'items in cluster: '+ str(len(clusters[id]))) for id in set(Y)])
plt.imshow(encodings_dict[0]['img'][0].reshape(416, 416, 3))
plt.show()
print('done')
