#!/bin/sh

g++ -O3 -march=native -DHIDDEN_COUNT=45 2b_trainNN.cpp -o tmp/2b_trainNN
g++ -O3 -march=native -DHIDDEN_COUNT=45 3b_predictNN.cpp -o tmp/3b_predictNN

python 1b_parseWithFeaturePreprocess.py data/spam_train.csv data/spam_test.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2b_trainNN tmp/featureMatrix tmp/labelMatrix weight_nn 60 -v
tmp/3b_predictNN tmp/testingFeatureMatrix weight_nn nn.csv 60
cp nn.csv submission/`date '+%Y%m%d_%H%M'`.csv

