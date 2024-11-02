## Bonus A

With batch size 128 and learning rate 0.001 and some guesses to architectural changes (additional linear layer), results passed 5% multiple times, but were still rather inconsistent (3~6%). However, compared with the original architecture's results, the loss values and their change across epochs were indicative of the model learning (loss decreasing across 10 epochs from ~300 down to 10). I also  found that test results were not the same when running the test script multiple times on the same model.  
So I had a look at the testing script and noticed that the y label encoding for a given class seemed to be inconsistent across runs (depending on the order in which classes were seen after shuffling of the testing data?).

- Fixed label encoding for ilabel in WikiArtDataset.__getitem__ by using a dictionary mapping arttype to idx (entries sorted alphabetically to assure consistency across training and testing)

After fixing label encoding, test results were consistent when re-testing the same model. Performance for the model as-is caps around 16% (1), which is the baseline for always predicting the most frequent arttype in the data (101/630 items in the test set).

To improve performance with minimal changes, I tried different layers and functions (additional linear layer of size 105*105/10, tanh, leakyrelu) and experimented with the hyper parameters. Once or twice, results were high for tanh but varied greatly. Decreasing the learning rate generally lead to loss continuing to decrease, but no improvement in performance.  
I landed on the following, which was not completely reliable either, but got varying and often times higher results than the base architecture (2):

Changes:  
- replaced Maxpool2d layer with nn.AdaptiveAvgPool2d((50,50)), also changes hidden layer size to 50*50 (prev: 105*105)
- swapped out relu for nn.Sigmoid()
- (changed attribute names: self.maxpool2d -> self.pool, self.relu -> self.activfunc for easy switching between architectures for comparison, added keyword mode to config/model/train function, pass mode='bonusA' in config for avgpool + sigmoid version)


### Condensed output  
(1) Code as-is with the fixed testing script:  
In epoch 0, loss = 4239.98828125  
In epoch 1, loss = 1162.042724609375  
...  
In epoch 19, loss = 1178.9503173828125  
Accuracy: 0.16031746566295624  

(2) Model with average pooling & sigmoid:  
In epoch 0, loss = 1267.143798828125  
In epoch 1, loss = 1221.037353515625  
...  
In epoch 19, loss = 1219.9281005859375  
Accuracy: 0.19365079700946808  

___

## Part 1

I tried accounting for the class imbalance by using nn.NLLLoss()'s weight parameter to weight the loss of a given class inverse to its frequency (the less frequent the class, the more is its loss weighed). I tried some different metrics (1-3), but none improved (and actually lowered) performance for either the base or the bonus part architectures from above.  
This leads me to believe that the weights work as intended, but the current architecture is not suited to actually learn the proper classification and therefore performs worse when it becomes impossible to exploit the class imbalance. In Part 2, I continued with weight metric 3.

Changes:  
- added label_counts attribute in WikiArtDataset: dict mapping arttype to its absolute frequency
- added class_weights calculation in train.py train() function & passed weight tensor to NLLLoss function's weight parameter

(1) weight_per_class_i = amount_training_samples / frequency_class_i_in_training / amount_classes_total
(2) weight_per_class_i = amount_training_samples / frequency_class_i_in_training  
(3) weight_per_class_i = 1 / frequency_class_i_in_training. 

### Example runs  

#### Weights (1)  

Base model (acc: 2,06%)  
In epoch 0, loss = 6005  
In epoch 1, loss = 1374  
...  
In epoch 19, loss = 1367  

bonusA model (acc: 5,71%)  
In epoch 0, loss = 1567  
In epoch 1, loss = 1476  
...  
In epoch 19, loss = 1510  

#### Weights (2)

Base model (acc: 2,06%)  
In epoch 0, loss = 7729  
In epoch 1, loss = 1356  
...  
In epoch 19, loss = 1374  

bonusA (acc: 7,94%)  
In epoch 0, loss = 1562  
In epoch 1, loss = 1442  
...  
In epoch 19, loss = 1469  

#### Weights (3)

Basic model (acc: 8,1%)
In epoch 0, loss = 8391  
In epoch 1, loss = 1370  
In epoch 19, loss = 1372  

bonusA (acc: 7,3%)  
In epoch 0, loss = 1630  
In epoch 1, loss = 1457  
In epoch 19, loss = 1428  


Literature: 
https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75, Oct 16
https://www.geeksforgeeks.org/handling-class-imbalance-in-pytorch/#2-class-weighting, Oct 18

___

## Part 2

### Autoencoder (class WikiArtPart2, in wikiart.py)

For the encoder I used two convolutional layers (3->9->3 channels, kernel size 5), each followed by a max pooling layer (kernel size & stride 2) to reduce the image size to 6,25% of the original size (from (3,416,416) to (3,104,104). The activation function between the two Conv2d/pooling pairs was relu.  
In the decoder, the encoder's structure was mirrored by two pairs consisting of a transposed convolutional layer followed by an upsampling layer (scale=2) to reconstruct the original input size. The activation function between the pairs was once again relu and the decoder's final layer employed the sigmoid function.  
Progress was measured by loss values (MSE_loss, initial value, change across epochs, approximate converging value) and plotting the encoded and decoded images, as well as comparison of the latter to the original image from the dataset.

Some experiments with increasing the amount of channels, adding more pairs of convolutional/pooling layers, varying the pooling type, or interspersing layers with more activation functions did not appear to improve performance further. Of my experiments, the structure above was the only one able to retrieve some of the original colours. Reducing the channel size below 3 at any point in the encoder made it impossible to retrieve proper colour values in decoding. 

Input images were also rescaled to range [0,1] via division by 255 just after reading the image in the WikiArtImage.get() call.

Performance as indicated by loss was fairly constant (initial: ~4, reaching and converging around ~1.8 after about 5 epochs; when run for 10 epochs) with the decoded image being easily recognisable/readable, but occasionally suffering in colour quality.  

### Autoencoder output

Original image  
![alt text](https://github.com/jshrdt/lt2326-h24-wa_modeling/blob/main/img4_original.png?raw=true)

Decoded image  
![alt text](https://github.com/jshrdt/lt2326-h24-wa_modeling/blob/main/img4_decoded.png?raw=true)

Encoded image  
![alt text](https://github.com/jshrdt/lt2326-h24-wa_modeling/blob/main/img4_encoded.png?raw=true)


### Clustering

First, I flattened the matrix of encoded 3D image (104,104,3) from the testset to then fit and apply sklearn's StandardScaler to all values. The scaled encodings I used to perform KMeans (n=27) clustering. Lastly, I applied PCA to the scaled encodings to reduce each image's array to a singular value in order to plot these PCA values (x-Axis) against the predicted KMeans-clusters (y-Axis). The graph I further colourcoded by the image's original gold label.

? Images aligned horizontally?

Images do not appear to cluster well along the x-values, but it is rather noticeable, that the gold label colours align horizontally. The colour spread not being entirely random implies that some information is still present in the encodings.

Personally, I found the graph was more interpretable when plotting the true class against the PCA value & keeping the clusters colour-coded. The resulting graph also displays the class imbalance from the dataset. Further, it shows the correlation between PCA value and true class - so perhaps clustering PCA values would have been the way to go.

[add imgs]

Literature:  
https://www.kaggle.com/code/dhanyajothimani/basic-visualization-and-clustering-in-python

---


## Part 3

To get style embeddings I initialised an embeddings layer with 27 random embeddings (one for every art style in the dataset), each of  the same size as the flattened artworks (3*416*416). Embeddings were then trained rather naively with the same auto encoder structure as in Part2. Training aimed to minimise MSE loss to the original image. I experiment with changing the structure, with no noticeable success.

For style transfer training, I froze the embeddings and concatenated input images with their corresponding style embedding (indexed via its art style, y value from the batcher) along axis 1/the channel dimension. This resulted in a new encoder input of size (batch_size, 6, 416, 416). 
Though embedding and transfer training share the same model, due to this difference in initial channel size, they each have their own initial Conv2d network, differing only in the expected in_channel size.

For style transfer, I implemented two loss functions: 
1) MSE loss for autoencoder output and original 'content' image
2) MSE loss for autoencoder output and style embedding
By combining these two losses, the model is trained to retain as much information as possible from both content and style input at the same time.


### Results

The altered image post style transfer still resembled the original image, but perhaps diverged from it more strongly than in part 2. Essentially I would describe the output here as almost grey-scaling or applying a sepia filter to the original content image, intensity of this effect varies. 

This result starts making sense when inspecting the outputs from the style embeddings training: In trying to generalise over many different paintings in a given art style, the embedding is pushed towards a uniform grey-beige image (sort of a middle ground for all possible pixel values). Though this does not exactly represent style in a meaningful way, it seems the transfer part is working as intended. 
This can be illustrated by modifying the transfer training loop to give more weight to the style loss (loss = content_loss + (4 * style_loss)).

I suspect two main factors for these results, in which the model could be improved: 
Unsuitable gold style representation during embeddings training & art style as a basis to attempt generalisation over. The former could be improved by setting the gold standard not as the original image, but some feature representation of it (from VGG for example). As for the latter, I'm not too sure what the goal for an art style representation ought to look like, it seems more intuitive to me to attempt a style transfer from a single painting onto another one, or to perhaps create embeddings for a certain artist. Paintings from the same genre probably diverge too much as to allow proper embeddings training with the methods I have used here.


Literature: 
Gates, L. A., Ecker, A. S., Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2414-2423. https://doi.org/10.1109/CVPR.2016.265

___

## Bonus B

