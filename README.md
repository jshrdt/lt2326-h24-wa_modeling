## Bonus A

- Increased batch size to 128 (prev: 32): due to the class imbalance, this might make it more likely for the model to see each class at least once per batch
- Added linear layer, linear15 with size ~1k (dim 104*104/10): intermediary stepping stone between jump from linear1 with ~10k to linear2 with 300 dimensions 
- Decreased learning rate to 0.001 (prev: 0.01): following intuition from what improved performance in assignment 1
- Decreased epochs, here: 12 (varied 10-15, prev: 20): loss seemed to bottom out (values ~5-20) around that point
- Decreased padding in conv2d/maxpool2d to 1 (prev: 2): again following intuition from assignment 1, changes hidden_size to 104*104

Results passed 5% multiple times, occasionally dipping below 5% accuracy. I assume the class imbalance to play a large role in this, and since absolute consistency did not seem to be reqcuired for this part, I stuck with these  changes. Though minimal, when compared with the original architecture's results, the loss values and their change across epochs is now indicative of the model being able to learn.

### Condensed output for wikiart_5perc.pth

$ python3 train.py  
Running...  
Gathering files for /scratch/lt2326-2926-h24/wikiart/train  
...............................finished  
In epoch 0, loss = 322.7892150878906  
In epoch 1, loss = 244.20724487304688  
In epoch 2, loss = 160.24156188964844  
In epoch 3, loss = 79.05674743652344  
In epoch 4, loss = 41.386409759521484  
In epoch 5, loss = 20.552532196044922  
In epoch 6, loss = 12.533958435058594  
In epoch 7, loss = 11.332183837890625  
In epoch 8, loss = 12.3660249710083  
In epoch 9, loss = 17.558002471923828  
In epoch 10, loss = 10.761187553405762  
In epoch 11, loss = 15.001067161560059  

$ python3 test.py  
Running...  
Gathering files for /scratch/lt2326-2926-h24/wikiart/test  
...............................finished    
Accuracy: 0.0634920671582222  
Confusion Matrix  
tensor([[ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  3.,  0.,  0.,
          1.,  0.,  1.,  2.,  2.,  1.,  1.,  0.,  0.,  0.,  2.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  1.,  1.,  2.,  0.,  0.,  2.,  0.,  0.,  0.,  3.,  0.,  1.,
          1.,  0.,  0.,  1.,  1.,  4.,  3.,  0.,  0.,  0.,  2.,  0.,  0.],
        [ 2.,  0.,  0.,  1.,  2.,  1.,  0.,  0.,  0.,  3.,  0.,  6.,  0.,  3.,
          2.,  0.,  5.,  0.,  0.,  1.,  6.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  3.,  0.,  0.,
          0.,  0.,  0.,  1.,  1.,  1.,  3.,  0.,  0.,  0.,  1.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  2.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  1.,  1.,  0.,  1.,  0.,  2.,  0.,  2.,  1.,  0.,
          2.,  0.,  0.,  2.,  0.,  2.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  1.,  0.,  1.,  2.,  0.,  0.,  1.,  0.,  1.,  1., 11.,  1.,  0.,
          3.,  0.,  1.,  6.,  1.,  0., 15.,  0.,  0.,  0.,  5.,  1.,  0.],
        [ 2.,  0.,  0.,  1.,  1.,  0.,  0.,  1.,  0.,  0.,  0., 13.,  1.,  0.,
          0.,  0.,  0.,  6.,  0.,  4., 12.,  0.,  0.,  0.,  9.,  1.,  0.],
        [ 0.,  0.,  0.,  1.,  9.,  0.,  0.,  1.,  0.,  4.,  0., 20.,  0.,  2.,
          2.,  0.,  1., 12.,  0.,  2., 41.,  0.,  0.,  0.,  6.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          1.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  3.,  2.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  5.,  0.,  0.,  0.,  0.,  2.,  0.,  5.,  0.,  0.,
          0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  3.,  0.,  1.,  0.,  0.,  2.,  0., 10.,  0.,  1.,
          1.,  0.,  1.,  4.,  0.,  4., 19.,  0.,  0.,  0.,  3.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 1.,  0.,  0.,  1.,  3.,  2.,  0.,  0.,  0.,  1.,  0., 18.,  2.,  5.,
          4.,  0.,  1.,  5.,  1.,  6., 25.,  0.,  2.,  0.,  4.,  1.,  0.],
        [ 1.,  0.,  0.,  0.,  2.,  0.,  1.,  0.,  1.,  1.,  0.,  8.,  0.,  0.,
          2.,  0.,  2.,  3.,  2.,  2.,  2.,  0.,  3.,  0.,  2.,  1.,  1.],
        [ 0.,  0.,  0.,  1.,  2.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,
          2.,  0.,  1.,  0.,  0.,  3.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  4.,  0.,  0.,  1.,  0.,  2.,  0.,  8.,  0.,  1.,
          0.,  0.,  0.,  6.,  1.,  3., 12.,  0.,  1.,  1.,  4.,  0.,  0.],
        [ 0.,  0.,  0.,  1.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          1.,  0.,  0.,  2.,  0.,  1.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  1.,  0.,  2.,  0.,  0.,
          0.,  0.,  0.,  1.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 1.,  0.,  0.,  0.,  3.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  1.,  9.,  0.,  0.,  0.,  2.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  1.,  1.,  0.,  4.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  2.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
          0.,  0.,  0.,  2.,  0.,  3.,  2.,  0.,  0.,  0.,  2.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
          0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.]])
__

## Part 1

- Account for imbalanced classes by using nn.NLLLoss()'s weight parameter to weight the loss per-class inverse to the class' frequency (https://medium.com/@zergtant/use-weighted-loss-function-to-solve-imbalanced-data-classification-problems-749237f38b75, Oct 16)

Changes:
- label_counts dict attribute added in WikiArtDataset: Absolute frequency of each arttype in data
- class_weights dict & weights tensor added in train.py train() function
- weights tensor of length=n_classes passed to NLLLoss function

___

## Part 2

___

## Part 3

___

## Bonus B

