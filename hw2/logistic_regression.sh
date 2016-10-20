#!/bin/sh

#g++ -O3 -march=native 2a_trainLinearRegression.cpp -o tmp/2a_trainLinearRegression
#g++ -O3 -march=native 3a_predictLinearRegression.cpp -o tmp/3a_predictLinearRegression

python 1b_parseWithFeaturePreprocess.py data/spam_train.csv data/spam_test.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
#tmp/2a_trainLinearRegression tmp/featureMatrix tmp/labelMatrix weight_linearRegression 49
#tmp/3a_predictLinearRegression tmp/testingFeatureMatrix weight_linearRegression linear_regression.csv 49

