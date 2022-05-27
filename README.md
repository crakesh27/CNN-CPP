# CNN-CPP
An extensible and modular implementation of Convolution Neural Networks in C++.

## Requirements
* C++ 11

## Building

The dataset used for training is MNIST handwritten digits dataste. It contains 60,000 training samples and 10,000 testing examples. You can download it from [MNIST link](http://yann.lecun.com/exdb/mnist/). Please extract all four files and place it in a separate folder named 'MNIST'.

To run, execute following commands
```
g++ load_image.cpp layers.cpp cnn.cpp -o test.out
./test.out
```

## Output

Accuracy around 92.47% when trained for 20 epochs with 10 batch size and 0.5 learning rate.

```
60000 images
60000 labels
Epoch : 1/20
Batch : 6000/6000 Training Loss: 0.323261
Time taken : 130.9434204102sec
Test Accuracy : 0.6905666590    Test Loss : 0.2095038593
Epoch : 2/20
Batch : 6000/6000 Training Loss: 0.1709278524
Time taken : 128.9757385254sec
Test Accuracy : 0.7898333073    Test Loss : 0.1530781686
Epoch : 3/20
Batch : 6000/6000 Training Loss: 0.1308346391
Time taken : 129.6773834229sec
Test Accuracy : 0.8605833054    Test Loss : 0.1087474748
Epoch : 4/20
Batch : 6000/6000 Training Loss: 0.1005350351
Time taken : 128.2032928467sec
Test Accuracy : 0.8832333088    Test Loss : 0.0921756253
Epoch : 5/20
Batch : 6000/6000 Training Loss: 0.0876044407
Time taken : 127.9117736816sec
Test Accuracy : 0.8980500102    Test Loss : 0.0819654241
Epoch : 6/20
Batch : 6000/6000 Training Loss: 0.0798420981
Time taken : 127.6298980713sec
Test Accuracy : 0.8978333473    Test Loss : 0.0818339363
Epoch : 7/20
Batch : 6000/6000 Training Loss: 0.0752588511
Time taken : 127.9868087769sec
Test Accuracy : 0.9021333456    Test Loss : 0.0782186687
Epoch : 8/20
Batch : 6000/6000 Training Loss: 0.0717870668
Time taken : 127.7466049194sec
Test Accuracy : 0.9084166884    Test Loss : 0.0727907568
Epoch : 9/20
Batch : 6000/6000 Training Loss: 0.0690624490
Time taken : 127.7570114136sec
Test Accuracy : 0.9107166529    Test Loss : 0.0722316727
Epoch : 10/20
Batch : 6000/6000 Training Loss: 0.0663161054
Time taken : 127.9231414795sec
Test Accuracy : 0.9190833569    Test Loss : 0.0643773302
Epoch : 11/20
Batch : 6000/6000 Training Loss: 0.0639779940
Time taken : 127.9275512695sec
Test Accuracy : 0.9213166833    Test Loss : 0.0632545203
Epoch : 12/20
Batch : 6000/6000 Training Loss: 0.0630218387
Time taken : 127.8605957031sec
Test Accuracy : 0.9217833281    Test Loss : 0.0625450537
Epoch : 13/20
Batch : 6000/6000 Training Loss: 0.0606022254
Time taken : 127.7766952515sec
Test Accuracy : 0.9232333302    Test Loss : 0.0612221025
Epoch : 14/20
Batch : 6000/6000 Training Loss: 0.0596275069
Time taken : 127.7988128662sec
Test Accuracy : 0.9236999750    Test Loss : 0.0606820211
Epoch : 15/20
Batch : 6000/6000 Training Loss: 0.0583804324
Time taken : 127.9415817261sec
Test Accuracy : 0.9226333499    Test Loss : 0.0616037995
Epoch : 16/20
Batch : 6000/6000 Training Loss: 0.0563948229
Time taken : 128.7188110352sec
Test Accuracy : 0.9245833158    Test Loss : 0.0605751537
Epoch : 17/20
Batch : 6000/6000 Training Loss: 0.0559208356
Time taken : 128.3111114502sec
Test Accuracy : 0.9254500270    Test Loss : 0.0596362650
Epoch : 18/20
Batch : 6000/6000 Training Loss: 0.0545428991
Time taken : 128.6391754150sec
Test Accuracy : 0.9259833097    Test Loss : 0.0595779866
Epoch : 19/20
Batch : 6000/6000 Training Loss: 0.0537633859
Time taken : 128.4654388428sec
Test Accuracy : 0.9280333519    Test Loss : 0.0578881949
Epoch : 20/20
Batch : 6000/6000 Training Loss: 0.0531267338
Time taken : 127.8627777100sec
Test Accuracy : 0.9269833565    Test Loss : 0.0591480248
10000 images
10000 labels
Test Accuracy : 0.9247999787    Test Loss : 0.0607771948
```

## Network Structure

Convolution neural network for the multi-classification.

Network flow:
Input-Convolution Layer-Pooling layer-FC1-FC2-Output

## Contribution
Contributions are always welcome! For implementing custom layers, you can just inherit `Layer` class within layers.cpp and implement all virtual functions. Few enhancements planned, implementation of softmax layer, corssentropy loss, adam optimizer, etc,.

## License
Distributed under the GPL-3.0 License. See `LICENSE.txt` for more information.