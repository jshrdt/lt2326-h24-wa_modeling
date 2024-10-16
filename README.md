## Bonus A
- Increased batch size to 128 (prev: 32): due to the class imbalance, this might make it more likely for the model to see each class at least once per batch
- Decreased padding in conv2d/maxpool2d to 1 (prev: 2): following intuition from assignment 1; changes hidden_size to 104*104
- Added linear layer, linear15 with size ~1k (dim 104*104/10): intermediary stepping stone between jump from linear1 with ~10k to linear2 with 300 dimensions 
- Decreased learning rate to 0.001 (prev: 0.01): again following intuition from what improved performance in assignment 1
- Decreased epochs to 10-15 (prev: 20): loss seemed to bottom out (values ~5-20) around that point

Model may still dip below 5% accuracy occasionally, but it seemed absolute reliability is not a recquirement for the bonus part. Compared with the loss values and changes across epochs, learning is certainly possible with the changes made. Especially learning rate & batch size appeared to have a big effect on performance.

### Sample output for wikiart_5perc.pth
Running...
Gathering files for /scratch/lt2326-2926-h24/wikiart/train
...............................finished
Starting epoch 0
100%|██████████████████████████████████████████████| 105/105 [00:40<00:00,  2.61it/s]
In epoch 0, loss = 322.7892150878906
Starting epoch 1
100%|██████████████████████████████████████████████| 105/105 [00:11<00:00,  8.80it/s]
In epoch 1, loss = 244.20724487304688
Starting epoch 2
100%|██████████████████████████████████████████████| 105/105 [00:12<00:00,  8.58it/s]
In epoch 2, loss = 160.24156188964844
Starting epoch 3
100%|██████████████████████████████████████████████| 105/105 [00:12<00:00,  8.51it/s]
In epoch 3, loss = 79.05674743652344
Starting epoch 4
100%|██████████████████████████████████████████████| 105/105 [00:12<00:00,  8.49it/s]
In epoch 4, loss = 41.386409759521484
Starting epoch 5
100%|██████████████████████████████████████████████| 105/105 [00:12<00:00,  8.41it/s]
In epoch 5, loss = 20.552532196044922
Starting epoch 6
100%|██████████████████████████████████████████████| 105/105 [00:12<00:00,  8.49it/s]
In epoch 6, loss = 12.533958435058594
Starting epoch 7
100%|██████████████████████████████████████████████| 105/105 [00:14<00:00,  7.35it/s]
In epoch 7, loss = 11.332183837890625
Starting epoch 8
100%|██████████████████████████████████████████████| 105/105 [00:16<00:00,  6.44it/s]
In epoch 8, loss = 12.3660249710083
Starting epoch 9
100%|██████████████████████████████████████████████| 105/105 [00:15<00:00,  6.73it/s]
In epoch 9, loss = 17.558002471923828
Starting epoch 10
100%|██████████████████████████████████████████████| 105/105 [00:14<00:00,  7.23it/s]
In epoch 10, loss = 10.761187553405762
Starting epoch 11
100%|██████████████████████████████████████████████| 105/105 [00:14<00:00,  7.40it/s]
In epoch 11, loss = 15.001067161560059
gussucju@GU.GU.SE@mltgpu:/srv/data/gussucju$ python3 test.py
Running...
Gathering files for /scratch/lt2326-2926-h24/wikiart/test
...............................finished
100%|█████████████████████████████████████████████| 630/630 [00:02<00:00, 248.16it/s]
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

___

## Part 2

___

## Part 3

___

## Bonus B

