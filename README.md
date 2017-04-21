# Usage Instructions:
### Train
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
* Run the train.sh in $ROOT/myself/fine-ResNet-101/train.sh to train the model
```
cd $ROOT
sh myself/fine-ResNet-101/train.sh
```

### Test

The test code is in $ROOT/myself/five_fold_cross_twoclass

* Run the five-fold-cross.py to test: in python terminal. 


