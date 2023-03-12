#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <numeric>
#include <vector>

#include "Util.h"

using namespace std;

class Perceptron {
public:
    Perceptron(int inputs, double bias = 1.0);
    vector<double> weights, m, v;
    double bias;
    double run(vector<double> x);
    void set_weights(vector<double> w_init);
    double sigmoid(double x);
};

#endif // PERCEPTRON_H
