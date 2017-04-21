### Usage Instructions:
#Train
* Clone this repository somewhere, let's refer to it as $ROOT
```
git clone https://github.com/longerping/cc-guardian.git
```
* compile the caffe and pycaffe.
```
cd $ROOT
make all 
make test 
make runtest 
make pycaffe
```
* Download the pre-trained model（https://github.com/KaimingHe/deep-residual-networks#models）
* Modify the values of two variables: pos_mult (specify the weight multiplier of a class) and pos_cid (the class number of the specified class) in $ROOT/src/caffe/layers/weighted_softmax_loss_layer.cpp
```
pos_mult_ = 12
pos_cid_ = 1
```
* Run the train.sh in $ROOT/myself/fine-ResNet-101/train.sh to train the model
```
cd $ROOT
sh myself/fine-ResNet-101/train.sh
```

#Test

The test code is in $ROOT/myself/five_fold_cross_twoclass

* Run the five-fold-cross.py to test: in python terminal. 


