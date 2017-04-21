#!/usr/bin/env sh

TOOLS=./build/tools

GLOG_logtostderr=1 $TOOLS/caffe train --solver=./myself/fine-ResNet-101/resnet_101_solver.prototxt --weights ./myself/fine-ResNet-101/ResNet-101-model.caffemodel -gpu 0,1,2,3

echo "Done."








