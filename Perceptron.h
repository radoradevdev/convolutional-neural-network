#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <numeric>
#include <vector>

#include "Util.h"

using namespace std;

class Perceptron {
public:
    Perceptron(int inputs, double bias = 1.0);

    double run(vector<double> x);
    void setWeights(vector<double> w_init);
    double sigmoid(double x);

    vector<double> weights, m, v;

private:
    double bias;
};

#endif // PERCEPTRON_H
