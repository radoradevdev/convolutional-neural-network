#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <time.h>
#include <vector>

#include "Util.h"
#include "Perceptron.h"

#define BETA1 0.9
#define BETA2 0.999
#define EPS 1e-7

using namespace std;

class FullyConnectedLayer {

public:
    FullyConnectedLayer(vector<int> layers, double bias = 1.0, bool adam = true,
                         double eta = 0.5);
    void set_weights(vector<vector<vector<double>>> w_init);
    void print_weights();
    vector<double> run(vector<double> x);
    vector<double> bp(vector<double> error);
    void gd();
    double Adam(double &m, double &v, double derivative);

    vector<int> layers;
    double bias;
    double eta;

    int back_iter;
    bool b_adam;

    vector<vector<Perceptron>> network;
    vector<vector<double>> values;
    vector<vector<double>> d;
    vector<vector<double>> loss_gradient;
};

#endif
