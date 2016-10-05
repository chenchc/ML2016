#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <algorithm>
using namespace std;

int FEATURE_COUNT;
int HIDDEN_COUNT = 200;
int NEG_COUNT = 50;
bool flag_validate = false;

double hiddenOutput(const vector<double> &feature, const vector<double> &hiddenWeight)
{
    double result = 0.0;
    
    for (int i = 0; i < FEATURE_COUNT; i++)
        result += feature[i] * hiddenWeight[i];

    result += hiddenWeight[FEATURE_COUNT];

    return result > 0 ? result : 0;
}

double predict(const vector<double> &feature, const vector<vector<double> > &weight, double bias)
{
    double predictLabel = 0.0;

    for (int i = 0; i < HIDDEN_COUNT; i++) {
        predictLabel += (i < NEG_COUNT ? -1.0 : 1.0) * hiddenOutput(feature, weight[i]);
    }
    predictLabel += bias;
   
    return predictLabel; 
}

int myrandom(int i)
{
    return rand() % i;
}

double train(vector<vector<double> > &weight, double &bias, const vector<vector<double> > &featureMatrix, 
    const vector<double> &labelMatrix, int iteration)
{
    const double ETA = 0.00000000005;
    double sum_squareError = 0.0;
    int size = flag_validate ? featureMatrix.size() * 2 / 3 : featureMatrix.size();

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
        // Pick an instance
        const int index = i % size;

        const vector<double> featureSet = newFeatureMatrix[index];
        double label = newLabelMatrix[index];

        // Gradient Descent
        vector<double> hiddenLayerOutput(HIDDEN_COUNT);
        double predictLabel = 0.0;

        for (int i = 0; i < HIDDEN_COUNT; i++) {
            hiddenLayerOutput[i] = hiddenOutput(featureSet, weight[i]);
            predictLabel += (i < NEG_COUNT ? -1.0 : 1.0) * hiddenLayerOutput[i];
        }
        predictLabel += bias;

        for (int i = 0; i < HIDDEN_COUNT; i++) {
            if (hiddenLayerOutput[i] <= 0)
                continue;
            for (int j = 0; j < FEATURE_COUNT; j++) {
                weight[i][j] += ETA * (i < NEG_COUNT ? -1.0 : 1.0) * (2 * (label - predictLabel) * featureSet[j]);
            }
            weight[i][FEATURE_COUNT] += ETA * (i < NEG_COUNT ? -1.0 : 1.0) * (2 * (label - predictLabel));
        }
        bias += ETA * (2 * (label - predictLabel));

        // Statistics
        sum_squareError += (label - predictLabel) * (label - predictLabel);
    }

    return sum_squareError / iteration;
}

double test(vector<vector<double> > &weight, double &bias, const vector<vector<double> > &featureMatrix, 
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
    vector<vector<double> > weight(HIDDEN_COUNT, 
        vector<double>(FEATURE_COUNT + 1));
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        for (int j = 0; j < FEATURE_COUNT; j++) {
            weight[i][j] = ((double)rand() / RAND_MAX - 0.5) / 1000;
        }
    }

    double bias = 0.0;

    double lastTestingMSESum = 6e23;
    double testingMSESum = 0.0;

    for (int i = 0; i < 300; i++) {
        double mse = train(weight, bias, featureMatrix, labelMatrix, 100000);
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
    for (int i = 0; i < HIDDEN_COUNT; i++) {
        for (int j = 0; j < FEATURE_COUNT + 1; j++) {
            fout_weight << weight[i][j] << endl;
        }
    }

    fout_weight << bias << endl;

    return 0;
}
