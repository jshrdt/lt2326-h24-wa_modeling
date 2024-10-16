## Bonus A


With batch size 128 and learning rate 0.001 and some guesses to architectural changes (additional linear layer), results passed 5% multiple times, but were still rather inconsistent (3~6%). However, compared with the original architecture's results, the loss values and their change across epochs were indicative of the model learning (loss decreasing across 10 epochs from ~300 down to 10). In addition, I noticed test results were not the same when running the test script multiple times on the same model.  
So I had a look at the testing script and noticed that the y label for a given class seemed to be inconsistent across runs (depending on the order in which classes were seen after shuffling of the testing data?).

- Fixed label encoding for ilabel in WikiArtDataset.__getitem__ by using a dictionary mapping arttypes to idx (entries sorted alphabetically to assure consistency across training and testing)

After fixing label encoding, test results were consistent when re-testing the same model. This alone brought accuracy to about 15%, even with the original parameters and architecture and loss being seemingly stagnant.


### Condensed output for wikiart_bonusA.pth
In epoch 0, loss = 4617.5654296875
In epoch 1, loss = 1135.1680908203125
...
In epoch 9, loss = 1175.9444580078125

Accuracy: 0.16031746566295624

___

## Part 1

- Account for imbalanced classes by using nn.NLLLoss()'s weight parameter to weight the loss per-class inverse to the class' frequency (https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75, Oct 16)

Changes:
- label_counts attribute added in WikiArtDataset: dict with arttype: absolute frequency
- class_weights dict & weights tensor added in train.py train() function
- weights tensor of length=n_classes passed to NLLLoss function

### Output
Running...
Gathering files for /scratch/lt2326-2926-h24/wikiart/train
...............................finished
Starting epoch 0
100%|██████████████████████████████████████████████| 418/418 [00:37<00:00, 11.21it/s]
In epoch 0, loss = 5407.64111328125
Starting epoch 1
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.60it/s]
In epoch 1, loss = 887.437255859375
Starting epoch 2
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 31.04it/s]
In epoch 2, loss = 852.6419677734375
Starting epoch 3
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 30.80it/s]
In epoch 3, loss = 852.1907348632812
Starting epoch 4
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 30.77it/s]
In epoch 4, loss = 848.8765869140625
Starting epoch 5
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 30.97it/s]
In epoch 5, loss = 864.7726440429688
Starting epoch 6
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 32.23it/s]
In epoch 6, loss = 861.0999755859375
Starting epoch 7
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.17it/s]
In epoch 7, loss = 855.6437377929688
Starting epoch 8
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.14it/s]
In epoch 8, loss = 854.3831787109375
Starting epoch 9
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.15it/s]
In epoch 9, loss = 857.7676391601562
gussucju@GU.GU.SE@mltgpu:/srv/data/gussucju$ python3 test.py 
Running...
Gathering files for /scratch/lt2326-2926-h24/wikiart/test
...............................finished
100%|█████████████████████████████████████████████| 630/630 [00:02<00:00, 270.51it/s]
Accuracy: 0.16031746566295624
Confusion Matrix

___

## Part 2

___

## Part 3

___

## Bonus B

