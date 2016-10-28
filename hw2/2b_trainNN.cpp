#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cfloat>
#include <fenv.h>
#include <assert.h>
using namespace std;

int FEATURE_COUNT;
bool flag_validate = false;
const int STOP_ITERATION = 165;

inline double sigmoid(double z) 
{
    double result = 1.0 / (1.0 + exp(-z));

    return result;
}

inline double relu(double z)
{
    return (z > 0.0) ? z : 0.0;
}

double predictWithDropout(vector<double> &output1, const vector<double> &feature, const vector<vector<double> > &weight0, const vector<double> &weight1, 
    const vector<bool> &dropout0, const vector<bool> &dropout1)
{
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        if (dropout1[i])
            continue;

        double z1 = 0.0;
        for (int j = 0; j < FEATURE_COUNT; j++) {
            if (dropout0[j])
                continue;
            z1 += feature[j] * weight0[i][j];
        }
        z1 += weight0[i][FEATURE_COUNT];

        output1[i] = relu(z1);
    }

    double z2 = 0.0;
    
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        if (dropout1[i])
            continue;
        z2 += output1[i] * weight1[i];
    }
    z2 += weight1[HIDDEN_COUNT];

    return sigmoid(z2);
}

double predict(vector<double> &output1, const vector<double> &feature, const vector<vector<double> > &weight0, const vector<double> &weight1)
{
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        double z1 = 0.0;
        for (int j = 0; j < FEATURE_COUNT; j++) {
            z1 += feature[j] * weight0[i][j] * (1 - DROPOUT0_PROB);
        }
        z1 += weight0[i][FEATURE_COUNT];

        output1[i] = relu(z1);
    }

    double z2 = 0.0;
    
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        z2 += output1[i] * weight1[i] * (1 - DROPOUT1_PROB);
    }
    z2 += weight1[HIDDEN_COUNT];

    return sigmoid(z2);
}

int myrandom(int i)
{
    return rand() % i;
}

double train(vector<vector<double> > &weight0, vector<double> &weight1, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix, int iteration)
{
    const double ETA = 0.00001;
    double sum_crossEntropy = 0.0;
    int size = flag_validate ? featureMatrix.size() * 4 / 5 : featureMatrix.size();

    vector<vector<double> > newFeatureMatrix;
    vector<double> newLabelMatrix;
    {
        vector<int> indexList;

        for (int i = 0; i < size; i++)
            indexList.push_back(i);
        random_shuffle(indexList.begin(), indexList.end(), myrandom);

        for (int i = 0; i < size; i++) {
            newFeatureMatrix.push_back(featureMatrix[i]);
            newLabelMatrix.push_back(labelMatrix[i]);
        }
    }

    for (int i = 0; i < iteration; i++) {
        for (int j = 0; j < size; j++) {
            // Pick an instance
            const int index = j;

            const vector<double> featureSet = newFeatureMatrix[index];
            double label = newLabelMatrix[index];

            // Generate dropout vector
            vector<bool> dropout0(FEATURE_COUNT);
            for (int i = 0; i < FEATURE_COUNT; i++) {
                dropout0[i] = ((double)rand() / RAND_MAX < DROPOUT0_PROB);
            }
            vector<bool> dropout1(HIDDEN_COUNT);
            for (int i = 0; i < HIDDEN_COUNT; i++) {
                dropout1[i] = ((double)rand() / RAND_MAX < DROPOUT1_PROB);
            }

            // Gradient Descent
            vector<double> output1(HIDDEN_COUNT);
            double predictLabel = predictWithDropout(output1, featureSet, weight0, weight1, dropout0, dropout1);
            double gradient_output2 = 2 * (label - predictLabel);

            vector<double> gradient_output1(HIDDEN_COUNT);
            for (int i = 0; i < HIDDEN_COUNT; i++) {
                if (dropout1[i])
                    continue;
                gradient_output1[i] = gradient_output2 * weight1[i];
                weight1[i] += ETA * gradient_output2 * output1[i];
            }
            weight1[HIDDEN_COUNT] += ETA * gradient_output2;
            
            for (int i = 0; i < HIDDEN_COUNT; i++) {
                if (dropout1[i])
                    continue;
                for (int j = 0; j < FEATURE_COUNT; j++) {
                    if (!dropout0[j] && output1[i] > 0.0)
                        weight0[i][j] += ETA * gradient_output1[i] * featureSet[j];
                }
                weight0[i][FEATURE_COUNT] += ETA * gradient_output1[i];
            }

            // Statistics
            if (label == 1.0)
                sum_crossEntropy += -label * log(predictLabel);
            else if (label == 0.0)
                sum_crossEntropy += -(1.0 - label) * log(1.0 - predictLabel);
            else
                exit(1);
        }
    }

    double ce = sum_crossEntropy / iteration / size;
    return ce;
}

double test(vector<vector<double> > &weight0, vector<double> &weight1, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix)
{
    double sum_crossEntropy = 0.0;

    for (int i = featureMatrix.size() * 4 / 5; i < featureMatrix.size(); i++) {
        vector<double> output1(HIDDEN_COUNT);
        double predictLabel = predict(output1, featureMatrix[i], weight0, weight1);
        double label = labelMatrix[i];

        if (label == 1.0)
            sum_crossEntropy += -label * log(predictLabel);
        else if (label == 0.0)
            sum_crossEntropy += -(1.0 - label) * log(1.0 - predictLabel);
        else
            exit(1);
    }

    return sum_crossEntropy / (featureMatrix.size() - featureMatrix.size() * 4 / 5);
}

double testAccuracy(vector<vector<double> > &weight0, vector<double> &weight1, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix)
{
    int count = 0;

    for (int i = featureMatrix.size() * 4 / 5; i < featureMatrix.size(); i++) {
        vector<double> output1(HIDDEN_COUNT);
        double predictLabel = predict(output1, featureMatrix[i], weight0, weight1);
        double label = labelMatrix[i];

        if (predictLabel < 0.5) {
            count += (label == 0) ? 1 : 0;
        }
        else {
            count += (label == 1) ? 1 : 0;
        }
    }

    return (double)count / (featureMatrix.size() - featureMatrix.size() * 4 / 5);
}

int main(int argc, char **argv)
{
    char *filename_featureMatrix = argv[1];
    char *filename_labelMatrix = argv[2];
    char *filename_weight = argv[3];
    FEATURE_COUNT = atoi(argv[4]);

    if (argc == 6)
        flag_validate = true;
   
    feenableexcept(FE_INVALID | FE_OVERFLOW);
    srand((unsigned int)time(NULL));

    // Load data
    vector<vector<double> > featureMatrix;
    vector<double> labelMatrix;
    
    ifstream fin_featureMatrix(filename_featureMatrix);
    while (!fin_featureMatrix.eof()) {
        vector<double> featureRow;
        for (int i = 0; i < FEATURE_COUNT; i++) {
            double element;

            fin_featureMatrix >> element;
            featureRow.push_back(element);
        }
        if (fin_featureMatrix.eof())
            break;
        featureMatrix.push_back(featureRow);
    }

    ifstream fin_labelMatrix(filename_labelMatrix);
    while (!fin_labelMatrix.eof()) {
        double element;

        fin_labelMatrix >> element;
        if (fin_labelMatrix.eof())
            break;
        labelMatrix.push_back(element);
    }
    
    // Train
    vector<vector<double> > weight0(
        HIDDEN_COUNT, 
        vector<double>(FEATURE_COUNT + 1, 0.0)
        );
    vector<double> weight1(HIDDEN_COUNT + 1, 0.0);

    for (int i = 0; i < HIDDEN_COUNT; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++ ) {
            weight0[i][j] = ((double)rand() / RAND_MAX - 0.5) * 2.0 * sqrt(6.0 / (FEATURE_COUNT + HIDDEN_COUNT));
        }
        weight1[i] = ((double)rand() / RAND_MAX - 0.5) * 8.0 * sqrt(6.0 / (HIDDEN_COUNT + 1));
    }

    double lastTestingCESum = 6e23;
    double testingCESum = 0.0;
    int stopIteration = flag_validate ? INT_MAX : STOP_ITERATION;

    for (int i = 0; i < stopIteration; i++) {
        double ce = train(weight0, weight1, featureMatrix, labelMatrix, 100);
        cout << "Epoch #" << i << ": Training Data CE=" << ce << endl;

        if (flag_validate) {
            double testingCE = test(weight0, weight1, featureMatrix, labelMatrix);
            testingCESum += testingCE;

            if (i % 10 == 9) {
                if (testingCESum > lastTestingCESum) {
                    cout << "Best testing CE=" << lastTestingCESum / 10 << endl;
                    cout << "Best training CE=" << ce << endl;
                    break;
                }

                // Output weight
                ofstream fout_weight(filename_weight);
                fout_weight.precision(20);
                for (int i = 0; i < HIDDEN_COUNT; i++) {
                    for (int j = 0; j < FEATURE_COUNT + 1; j++) {
                        fout_weight << weight0[i][j] << endl;
                    }
                }
                for (int i = 0; i < HIDDEN_COUNT + 1; i++) {
                    fout_weight << weight1[i] << endl;
                }

                lastTestingCESum = testingCESum;
                testingCESum = 0.0;
            }

            double testingAccuracy = testAccuracy(weight0, weight1, featureMatrix, labelMatrix);
            cout << "Epoch #" << i << ": Testing Data CE=" << testingCE << " Accuracy=" << testingAccuracy << endl;
        }
    }

    return 0;
}
