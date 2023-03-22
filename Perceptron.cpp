#include "Perceptron.h"

// Return a new Perceptron object with the specified number of inputs (+1 for
// the bias).
Perceptron::Perceptron(int inputs, double bias) {

    this->bias = bias;
    weights.resize(inputs + 1);
    m.resize(inputs + 1, 0);
    v.resize(inputs + 1, 0);
    generate(weights.begin(), weights.end(), Util::frand);
}

// Run the perceptron. x is a vector with the input values.
double Perceptron::run(vector<double> x) {
    x.push_back(bias);
    double sum = inner_product(x.begin(), x.end(), weights.begin(), (double)0.0);
    return sigmoid(sum);
}

// Set the weights. w_init is a vector with the weights.
void Perceptron::setWeights(vector<double> w_init) { weights = w_init; }

// Evaluate the sigmoid function for the floating point input x.
double Perceptron::sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }
