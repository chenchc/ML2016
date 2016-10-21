#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cfloat>
#include <fenv.h>
using namespace std;

int FEATURE_COUNT;
bool flag_validate = false;
const int STOP_ITERATION = 165;

inline double sigmoid(double z) 
{
    double result = 1.0 / (1.0 + exp(-z));

    return result;
}

double predict(const vector<double> &feature, const vector<double> &weight, double bias)
{
    double result = 0.0;
    
    for (int i = 0; i < FEATURE_COUNT; i++)
        result += feature[i] * weight[i];

    result += bias;

    return sigmoid(result);
}

int myrandom(int i)
{
    return rand() % i;
}

double train(vector<double> &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix, int iteration)
{
    const double ETA = 0.000005;
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

            // Gradient Descent
            double predictLabel = predict(featureSet, weight, bias);
            for (int i = 0; i < FEATURE_COUNT; i++) {
                weight[i] += ETA * (2 * (label - predictLabel) * featureSet[i]);
            }
            bias += ETA * (2 * (label - predictLabel));

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

double test(vector<double> &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix)
{
    double sum_crossEntropy = 0.0;

    for (int i = featureMatrix.size() * 4 / 5; i < featureMatrix.size(); i++) {
        double predictLabel = predict(featureMatrix[i], weight, bias);
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

double testAccuracy(vector<double> &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix)
{
    int count = 0;

    for (int i = featureMatrix.size() * 4 / 5; i < featureMatrix.size(); i++) {
        double predictLabel = predict(featureMatrix[i], weight, bias);
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

    double lastTestingCESum = 6e23;
    double testingCESum = 0.0;
    int stopIteration = flag_validate ? INT_MAX : STOP_ITERATION;

    for (int i = 0; i < stopIteration; i++) {
        double ce = train(weight, bias, featureMatrix, labelMatrix, 1000);
        cout << "Epoch #" << i << ": Training Data CE=" << ce << endl;

        if (flag_validate) {
            double testingCE = test(weight, bias, featureMatrix, labelMatrix);
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
                for (int i = 0; i < FEATURE_COUNT; i++)
                    fout_weight << weight[i] << endl;

                fout_weight << bias << endl;

                lastTestingCESum = testingCESum;
                testingCESum = 0.0;
            }

            double testingAccuracy = testAccuracy(weight, bias, featureMatrix, labelMatrix);
            cout << "Epoch #" << i << ": Testing Data CE=" << testingCE << " Accuracy=" << testingAccuracy << endl;
        }
    }

    // Output weight
    /*
    ofstream fout_weight(filename_weight);
    fout_weight.precision(20);
    for (int i = 0; i < FEATURE_COUNT; i++)
        fout_weight << weight[i] << endl;

    fout_weight << bias << endl;
    */

    return 0;
}
