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
100%|██████████████████████████████████████████████| 418/418 [00:38<00:00, 10.92it/s]
In epoch 0, loss = 1490.2838134765625
Starting epoch 1
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.76it/s]
In epoch 1, loss = 659.7559814453125
Starting epoch 2
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 30.99it/s]
In epoch 2, loss = 516.181396484375
Starting epoch 3
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 29.94it/s]
In epoch 3, loss = 268.4538269042969
Starting epoch 4
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 29.95it/s]
In epoch 4, loss = 174.8949737548828
Starting epoch 5
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 29.93it/s]
In epoch 5, loss = 416.5644226074219
Starting epoch 6
100%|██████████████████████████████████████████████| 418/418 [00:13<00:00, 30.05it/s]
In epoch 6, loss = 185.43495178222656
Starting epoch 7
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 33.00it/s]
In epoch 7, loss = 79.78329467773438
Starting epoch 8
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 33.76it/s]
In epoch 8, loss = 176.1267852783203
Starting epoch 9
100%|██████████████████████████████████████████████| 418/418 [00:12<00:00, 34.06it/s]
In epoch 9, loss = 73.84375
gussucju@GU.GU.SE@mltgpu:/srv/data/gussucju$ python3 test.py 
Running...
Gathering files for /scratch/lt2326-2926-h24/wikiart/test
...............................finished
100%|█████████████████████████████████████████████| 630/630 [00:02<00:00, 252.54it/s]
Accuracy: 0.1746031790971756
Confusion Matrix

___

## Part 2

___

## Part 3

___

## Bonus B

