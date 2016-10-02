#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
using namespace std;

const int FEATURE_COUNT = 18 * 9;

double predict(const vector<double> &feature, const vector<double> &weight, double bias)
{
    double result = 0.0;
    
    for (int i = 0; i < FEATURE_COUNT; i++)
        result += feature[i] * weight[i];

    result += bias;

    return result;
}

double train(vector<double> &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix, int iteration)
{
    const double ETA = 0.000000001;
    double sum_squareError = 0.0;

    for (int i = 0; i < iteration; i++) {
        // Randomly pick an instance
        const int index = rand() % featureMatrix.size();
        const vector<double> featureSet = featureMatrix[index];
        double label = labelMatrix[index];

        // Gradient Descent
        double predictLabel = predict(featureSet, weight, bias);
        for (int i = 0; i < FEATURE_COUNT; i++) {
            weight[i] += ETA * (label - predictLabel) * featureSet[i];
        }
        bias += ETA * (label - predictLabel);

        // Statistics
        sum_squareError += (label - predictLabel) * (label - predictLabel);
    }

    return sum_squareError / iteration;
}

int main(int argc, char **argv)
{
    char *filename_featureMatrix = argv[1];
    char *filename_labelMatrix = argv[2];
    char *filename_weight = argv[3];
    
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
    vector<double> weight(FEATURE_COUNT, 0.0);
    double bias = 0.0;

    for (int i = 0; i < 100; i++) {
        double mse = train(weight, bias, featureMatrix, labelMatrix, 1000000);
        cout << "Epoch #" << i << ": MSE=" << mse << endl;
    }

    // Output weight
    ofstream fout_weight(filename_weight);
    for (int i = 0; i < FEATURE_COUNT; i++)
        fout_weight << weight[i] << endl;

    fout_weight << bias << endl;

    return 0;
}
