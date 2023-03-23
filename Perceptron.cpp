#include "Perceptron.h"

Perceptron::Perceptron(int inputs, double bias) {

    _bias = bias;
    _weights.resize(inputs + 1); // vector with inputs + 1 elements
    _past_gradient.resize(inputs + 1, 0); // vector with inputs + 1 elements, all 0s
    _squared_gradient.resize(inputs + 1, 0); // vector with inputs + 1 elements, all 0s

    // vector with inputs + 1 elements, all random floating-point numbers
    generate(_weights.begin(), _weights.end(), Util::frand);
}

double Perceptron::run(vector<double> x) {
    x.push_back(_bias);

    // Returns the result of accumulating 0.0 with the inner products of the pairs
    // formed by the elements of two ranges starting at x.begin() and _weights.begin().
    double sum = inner_product(x.begin(), x.end(), _weights.begin(), (double)0.0);
    return sigmoid(sum);
}

void Perceptron::setWeights(vector<double> init_weights) {
    _weights = init_weights;
}

double Perceptron::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
