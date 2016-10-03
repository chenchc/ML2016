#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
using namespace std;

int FEATURE_COUNT;
bool flag_validate = false;

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
    const double ETA = 0.0000000005;
    const double LAMBDA = 0.5;
    double sum_squareError = 0.0;

    for (int i = 0; i < iteration; i++) {
        // Randomly pick an instance
        const int index = flag_validate ? 
            rand() % (featureMatrix.size() * 2 / 3) : 
            rand() % featureMatrix.size();

        const vector<double> featureSet = featureMatrix[index];
        double label = labelMatrix[index];

        // Gradient Descent
        double predictLabel = predict(featureSet, weight, bias);
        for (int i = 0; i < FEATURE_COUNT; i++) {
            weight[i] += ETA * (2 * (label - predictLabel) * featureSet[i] - 2 * LAMBDA * (weight[i] > 0.0 ? 1.0 : (weight[i] < 0.0 ? -1.0 : 0.0)));
        }
        bias += ETA * (2 * (label - predictLabel));

        // Statistics
        sum_squareError += (label - predictLabel) * (label - predictLabel);
    }

    return sum_squareError / iteration;
}

double test(vector<double> &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix)
{
    double sum_squareError = 0.0;

    for (int i = featureMatrix.size() * 2 / 3; i < featureMatrix.size(); i++) {
        double predictLabel = predict(featureMatrix[i], weight, bias);
        double label = labelMatrix[i];

        sum_squareError += (label - predictLabel) * (label - predictLabel);
    }

    return sum_squareError / (featureMatrix.size() / 3);
}

int main(int argc, char **argv)
{
    char *filename_featureMatrix = argv[1];
    char *filename_labelMatrix = argv[2];
    char *filename_weight = argv[3];
    FEATURE_COUNT = atoi(argv[4]);

    if (argc == 6)
        flag_validate = true;
    
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

    double lastTestingMSESum = 6e23;
    double testingMSESum = 0.0;

    for (int i = 0; i < 700; i++) {
        double mse = train(weight, bias, featureMatrix, labelMatrix, 1000000);
        cout << "Epoch #" << i << ": Training Data MSE=" << mse << endl;

        if (flag_validate) {
            double testingMSE = test(weight, bias, featureMatrix, labelMatrix);

            if (i % 50 == 49) {
                if (testingMSESum > lastTestingMSESum) {
                    cout << "Best MSE=" << lastTestingMSESum / 50 << endl;
                    break;
                }
                lastTestingMSESum = testingMSESum;
                testingMSESum = 0.0;
            }
            testingMSESum += testingMSE;
            cout << "Epoch #" << i << ": Testing Data MSE=" << testingMSE  << endl;
        }
    }

    // Output weight
    ofstream fout_weight(filename_weight);
    fout_weight.precision(20);
    for (int i = 0; i < FEATURE_COUNT; i++)
        fout_weight << weight[i] << endl;

    fout_weight << bias << endl;

    return 0;
}
