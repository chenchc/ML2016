#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <cfloat>
#include <fenv.h>
#include <assert.h>
using namespace std;

int FEATURE_COUNT;

inline double sigmoid(double z) 
{
    return 1 / (1 + exp(-z));
}

inline double relu(double z)
{
    return (z > 0.0) ? z : 0.0;
}

double predict(vector<double> &output1, const vector<double> &feature, const vector<vector<double> > &weight0, const vector<double> &weight1)
{
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        double z1 = 0.0;
        for (int j = 0; j < FEATURE_COUNT; j++) {
            z1 += feature[j] * weight0[i][j];
        }
        z1 += weight0[i][FEATURE_COUNT];

        output1[i] = relu(z1);
    }

    double z2 = 0.0;
    
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        z2 += output1[i] * weight1[i];
    }
    z2 += weight1[HIDDEN_COUNT];

    return sigmoid(z2);
}

int myrandom(int i)
{
    return rand() % i;
}

int main(int argc, char **argv)
{
    char *filename_testingFeatureMatrix = argv[1];
    char *filename_weight = argv[2];
    char *filename_submission = argv[3];
    FEATURE_COUNT = atoi(argv[4]);

    feenableexcept(FE_INVALID | FE_OVERFLOW);

    // Load data
    vector<vector<double> > testingFeatureMatrix;
    vector<vector<double> > weight0(
        HIDDEN_COUNT,
        vector<double>(FEATURE_COUNT + 1)
    );
    vector<double> weight1(HIDDEN_COUNT + 1);
    
    ifstream fin_testingFeatureMatrix(filename_testingFeatureMatrix);
    while (!fin_testingFeatureMatrix.eof()) {
        vector<double> featureRow;
        for (int i = 0; i < FEATURE_COUNT; i++) {
            double element;

            fin_testingFeatureMatrix >> element;
            featureRow.push_back(element);
        }
        if (fin_testingFeatureMatrix.eof())
            break;
        testingFeatureMatrix.push_back(featureRow);
    }
    
    ifstream fin_weight(filename_weight);
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        for (int j = 0; j < FEATURE_COUNT + 1; j++) {
            double element;
            fin_weight >> element;
            weight0[i][j] = element;
        }
    }
    for (int i = 0; i < HIDDEN_COUNT + 1; i++) {
        double element;
        fin_weight >> element;
        weight1[i] = element;
    }
    
    // Predict
    vector<int> predictLabel;

    for (int i = 0; i < testingFeatureMatrix.size(); i++) {
        const vector<double> &feature = testingFeatureMatrix[i];
        vector<double> output1(HIDDEN_COUNT);
        double label = predict(output1, feature, weight0, weight1);
        if (label > 0.5)
            predictLabel.push_back(1);
        else
            predictLabel.push_back(0);
    }

    // Output submission
    ofstream fout_submission(filename_submission); 
    fout_submission << "id,label" << endl;                               
    for (int i = 0; i < predictLabel.size(); i++)                        
        fout_submission << i + 1 << "," << predictLabel[i] << endl; 

    return 0;
}
