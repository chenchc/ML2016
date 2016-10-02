#!/bin/sh

g++ -O3 -march=native 2a_trainLinearRegression.cpp -o tmp/2a_trainLinearRegression

python 1_parse.py data/train.csv data/test_X.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2a_trainLinearRegression tmp/featureMatrix tmp/labelMatrix weight_linearRegression

