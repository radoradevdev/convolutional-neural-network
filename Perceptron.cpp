#include "Perceptron.h"

Perceptron::Perceptron(int inputs, double bias) {

    _bias = bias;
    weights.resize(inputs + 1);
    past_gradient.resize(inputs + 1, 0);
    squared_gradient.resize(inputs + 1, 0);
    generate(weights.begin(), weights.end(), Util::frand);
}

double Perceptron::run(vector<double> x) {
    x.push_back(_bias);
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
    return sigmoid(sum);
}

void Perceptron::setWeights(vector<double> init_weights) { weights = init_weights; }

double Perceptron::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
