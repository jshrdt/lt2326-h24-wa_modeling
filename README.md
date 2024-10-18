## Bonus A

With batch size 128 and learning rate 0.001 and some guesses to architectural changes (additional linear layer), results passed 5% multiple times, but were still rather inconsistent (3~6%). However, compared with the original architecture's results, the loss values and their change across epochs were indicative of the model learning (loss decreasing across 10 epochs from ~300 down to 10). I also  found that test results were not the same when running the test script multiple times on the same model.  
So I had a look at the testing script and noticed that the y label encoding for a given class seemed to be inconsistent across runs (depending on the order in which classes were seen after shuffling of the testing data?).

- Fixed label encoding for ilabel in WikiArtDataset.__getitem__ by using a dictionary mapping arttype to idx (entries sorted alphabetically to assure consistency across training and testing)

After fixing label encoding, test results were consistent when re-testing the same model.
Performance for the model as-is caps around 16% (1), which is the baseline for always predicting the most frequent arttype in the data (101/630 items in the test set).

To improve performance with minimal changes, I tried different layers and functions (additional linear layer of size 105*105/10, tanh, leakyrelu) and experimented with the hyper parameters. Once or twice, results were high for tanh but varied greatly. Decreasing the learning rate generally lead to los continuing to decrease, but no improvement in performance.
I landed on the following (2), which was not completely reliable either, but got varying and often times higher results than the base architecture:

- replaced maxpool2d layer with nn.AdaptiveAvgPool2d((50,50)), also changes hidden layer size to 50*50 (prev: 105*105)
- swapped out relu for nn.Sigmoid()
- (changed attribute names: self.maxpool2d -> self.pool, self.relu -> self.activfunc for easy switching between architectures for comparison, added keyword mode to config/model/train function, pass mode='bonusA' in config for avgpool + sigmoid)


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

I tried accounting for the class imbalance by using nn.NLLLoss()'s weight parameter to weight the loss per-class inverse to the class' frequency. I tried some different metrics (1-3), but none improved (and actually lowered) performance for either of the architectures from the bonus part above.  
This leads me to believe that the weights might work as intended, but the current architecture is not suited to actually learn the proper classification and therefore has worse results when exploiting the class imbalance becomes impossible.

Changes:
- added label_counts attribute in WikiArtDataset: dict mapping arttype to its absolute frequency
- added class_weights tensor calculation in train.py train() function & passed weights to NLLLoss function's weight parameter

(1) weight_per_class_i = amount_training_samples / frequency_class_i_in_training / amount_classes_total
(2) weight_per_class_i = amount_training_samples / frequency_class_i_in_training  
(3) weight_per_class_i = 1 / frequency_class_i_in_training. 

(https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75, Oct 16)
(https://www.geeksforgeeks.org/handling-class-imbalance-in-pytorch/#2-class-weighting, Oct 18)

### Example runs
 basic architecture and weights 1
In epoch 0, loss = 6005.33935546875
In epoch 1, loss = 1374.005859375
...
In epoch 19, loss = 1367.11962890625
Accuracy: 0.02063492126762867 

bonusA and weights 1
In epoch 0, loss = 1567.849853515625
In epoch 1, loss = 1476.540771484375
...
In epoch 19, loss = 1510.24072265625
Accuracy: 0.05714285746216774

bonusA and weights 2
In epoch 0, loss = 1562.4427490234375
In epoch 1, loss = 1442.5166015625

In epoch 19, loss = 1469.7078857421875
Accuracy: 0.0793650820851326

Basic and 2
In epoch 0, loss = 7729.8564453125
In epoch 1, loss = 1356.2374267578125
In epoch 19, loss = 1374.14111328125
Accuracy: 0.02063492126762867

Basic and w3
In epoch 0, loss = 8391.287109375
In epoch 1, loss = 1370.884033203125
In epoch 19, loss = 1372.5889892578125
Accuracy: 0.08095238357782364

Bonus w3
In epoch 0, loss = 1630.231689453125
In epoch 1, loss = 1457.8216552734375
In epoch 19, loss = 1428.0557861328125
Accuracy: 0.07301587611436844
___

## Part 2

___

## Part 3

___

## Bonus B

