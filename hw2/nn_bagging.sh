#!/bin/bash

g++ -O3 -march=native -DHIDDEN_COUNT=60 -DDROPOUT0_PROB=0.0 -DDROPOUT1_PROB=0.2 2b_trainNN.cpp -o tmp/2b_trainNN
g++ -O3 -march=native -DHIDDEN_COUNT=60 -DDROPOUT0_PROB=0.0 -DDROPOUT1_PROB=0.2 3c_predictNNBagging.cpp -o tmp/3c_predictNNBagging

python 1b_parseWithFeaturePreprocess.py data/spam_train.csv data/spam_test.csv tmp/featureMatrix tmp/labelMatrix tmp/testingFeatureMatrix
for (( i=0; i<8; i=i+1 )); do
    python 1.5_sampling.py tmp/featureMatrix tmp/labelMatrix tmp/featureMatrix_${i} tmp/labelMatrix_${i} &
done
wait
for (( i=0; i<8; i=i+1 )); do
    tmp/2b_trainNN tmp/featureMatrix_${i} tmp/labelMatrix_${i} weight_nnbagging_${i} 60 -v &
done
wait
tmp/3c_predictNNBagging tmp/testingFeatureMatrix weight_nnbagging nnbagging.csv 60 8
cp nnbagging.csv submission/`date '+%Y%m%d_%H%M'`.csv

