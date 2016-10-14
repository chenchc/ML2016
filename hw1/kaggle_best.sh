#!/bin/sh

g++ -O3 -march=native 2c_trainNN.cpp -o tmp/2c_trainNN
g++ -O3 -march=native 3c_predictNN.cpp -o tmp/3c_predictNN

python 1b_parseWithFeatureSelection.py data/train.csv data/test_X.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2c_trainNN tmp/featureMatrix tmp/labelMatrix weight_NN 49
tmp/3c_predictNN tmp/testingFeatureMatrix weight_NN kaggle_best.csv 49

