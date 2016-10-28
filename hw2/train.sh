#!/bin/sh

g++ -O3 -march=native 2a_trainLogisticRegression.cpp -o tmp/2a_trainLogisticRegression
g++ -O3 -march=native 3a_predictLogisticRegression.cpp -o tmp/3a_predictLogisticRegression

python 1b_parseWithFeaturePreprocess.py ${1} dummy.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/2a_trainLogisticRegression tmp/featureMatrix tmp/labelMatrix $2 60 -v

