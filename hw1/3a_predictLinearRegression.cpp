#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
using namespace std;

int FEATURE_COUNT;

double predict(const vector<double> &feature, const vector<double> &weight, double bias)
{
    double result = 0.0;
    
    for (int i = 0; i < FEATURE_COUNT; i++)
        result += feature[i] * weight[i];

    result += bias;

    return result;
}

int main(int argc, char **argv)
{
    char *filename_testingFeatureMatrix = argv[1];
    char *filename_weight = argv[2];
    char *filename_submission = argv[3];
    FEATURE_COUNT = atoi(argv[4]);
    
    // Load data
    vector<vector<double> > testingFeatureMatrix;
    vector<double> weight;
    double bias;
    
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
    for (int i = 0; i < FEATURE_COUNT; i++) {
        double element;

        fin_weight >> element;
        weight.push_back(element);
    }
    fin_weight >> bias;
    
    // Predict
    vector<double> predictLabel;

    for (int i = 0; i < testingFeatureMatrix.size(); i++) {
        const vector<double> &feature = testingFeatureMatrix[i];
        double label = predict(feature, weight, bias);
        label = label < 0.0 ? 0.0 : label;
        predictLabel.push_back(label);
    }

    // Output submission
    ofstream fout_submission(filename_submission);
    fout_submission.precision(20);
    fout_submission << "id,value" << endl;
    for (int i = 0; i < predictLabel.size(); i++)
        fout_submission << "id_" << i << "," << predictLabel[i] << endl;

    return 0;
}
