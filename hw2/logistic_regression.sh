#!/bin/sh

g++ -O3 -march=native 2a_trainLogisticRegression.cpp -o tmp/2a_trainLogisticRegression
g++ -O3 -march=native 3a_predictLogisticRegression.cpp -o tmp/3a_predictLogisticRegression

python 1b_parseWithFeaturePreprocess.py data/spam_train.csv data/spam_test.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2a_trainLogisticRegression tmp/featureMatrix tmp/labelMatrix weight_logisticRegression 60 -v
tmp/3a_predictLogisticRegression tmp/testingFeatureMatrix weight_logisticRegression logistic_regression.csv 60
cp logistic_regression.csv submission/`date '+%Y%m%d_%H%M'`.csv

