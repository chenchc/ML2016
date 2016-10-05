#!/bin/sh

g++ -O3 -march=native 2a_trainLinearRegression.cpp -o tmp/2a_trainLinearRegression
g++ -O3 -march=native 3a_predictLinearRegression.cpp -o tmp/3a_predictLinearRegression

python 1b_parseWithFeatureSelection.py data/train.csv data/test_X.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2a_trainLinearRegression tmp/featureMatrix tmp/labelMatrix weight_linearRegression 49 -v

