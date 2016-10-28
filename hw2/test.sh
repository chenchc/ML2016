#!/bin/sh

g++ -O3 -march=native 2a_trainLogisticRegression.cpp -o tmp/2a_trainLogisticRegression
g++ -O3 -march=native 3a_predictLogisticRegression.cpp -o tmp/3a_predictLogisticRegression

python 1b_parseWithFeaturePreprocess.py dummy.csv $2 tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
tmp/3a_predictLogisticRegression tmp/testingFeatureMatrix $1 $3 60

