#include "FullyConnectedLayer.h"

FullyConnectedLayer::FullyConnectedLayer(
        vector<int> layers,
        double bias,
        bool adam,
        double eta
        ) {

    srand(time(NULL));

    _sublayers = layers;
    _bias = bias;
    _eta = eta;
    _back_iter = 0;
    _b_adam = adam;

    for (int i = 0; i < (int)layers.size(); i++) {
        _values.push_back(vector<double>(layers[i], 0.0));
        _d.push_back(vector<double>(layers[i], 0.0));
        _loss_gradient.push_back(vector<double>(layers[i], 0.0));
        _network.push_back(vector<Perceptron>());
        if (i > 0) { // network[0] is the input layer, so it doesn't have neurons
            for (int j = 0; j < layers[i]; j++) {
                _network[i].push_back(Perceptron(layers[i - 1], bias));
            }
        }
    }
}

void FullyConnectedLayer::setWeights(vector<vector<vector<double>>> weights) {
    for (int i = 0; i < (int)weights.size(); i++) {
        for (int j = 0; j < (int)weights[i].size(); j++) {
            _network[i + 1][j].setWeights(weights[i][j]);
        }
    }
}

void FullyConnectedLayer::printWeights() {
    cout << endl;
    for (int i = 1; i < (int)_network.size(); i++) {
        for (int j = 0; j < _sublayers[i]; j++) {
            cout << "Layer " << i + 1 << " Neuron " << j << ": ";
            for (auto &it : _network[i][j]._weights)
                cout << it << "   ";
            cout << endl;
        }
    }
    cout << endl;
}

vector<double> FullyConnectedLayer::fwd(vector<double> data) {

    _values[0] = data;
    for (int i = 1; i < (int)_network.size(); i++) {
        for (int j = 0; j < _sublayers[i]; j++) {
            _values[i][j] = _network[i][j].run(_values[i - 1]);
        }
    }
    return _values.back();
}

double FullyConnectedLayer::calcAdam(
        double &past_gradient,
        double &squared_gradient,
        double derivative
        ) {

    int t = _back_iter;
    double dx = derivative;

    past_gradient = BETA1 * past_gradient + (1 - BETA1) * dx;
    double mt = past_gradient / (1 - (double)pow(BETA1, t));
    squared_gradient = BETA2 * squared_gradient + (1 - BETA2) * ((double)pow(dx, 2));
    double vt = squared_gradient / (1 - (double)pow(BETA2, t));
    double delta = _eta * mt / (sqrt(vt) + EPS);

    return delta;
}

void FullyConnectedLayer::applyGradientDescent() {
    for (int i = 1; i < (int)_network.size(); i++)
        for (int j = 0; j < _sublayers[i]; j++)
            for (int k = 0; k < _sublayers[i - 1] + 1; k++) {
                double delta;
                if (k == _sublayers[i - 1]) {
                    delta = _eta * _d[i][j];
                } else {

                    double dw = _d[i][j] * _values[i - 1][k]; // dw = dscore * X

                    if (!_b_adam) {
                        delta = _eta * dw; // learning rate * dw
                    } else {
                        delta = calcAdam(_network[i][j]._past_gradient[k], _network[i][j]._squared_gradient[k], dw);
                    }
                }
                _network[i][j]._weights[k] += delta;
            }
}

vector<double> FullyConnectedLayer::bp(vector<double> error_gradient) {

    _back_iter++; // Updates adam result

    vector<double> outputs = _values.back();

    // Calculate the output error
    for (int i = 0; i < (int)outputs.size(); i++) {
        _d.back()[i] = outputs[i] * (1 - outputs[i]) * (error_gradient[i]);
    }

    // Calculate the error term of each unit on each layer
    for (int i = ((int)_network.size()) - 2; i > 0; i--) {

        for (int h = 0; h < (int)_network[i].size(); h++) {

            double fwd_error = 0.0;

            for (int k = 0; k < _sublayers[i + 1]; k++) {
                fwd_error += _network[i + 1][k]._weights[h] * _d[i + 1][k];
            }

            _loss_gradient[i][h] = fwd_error;
            _d[i][h] = _values[i][h] * (1 - _values[i][h]) * fwd_error;
        }
    }

    applyGradientDescent();

    return _loss_gradient[1]; // Result from the first layer (not the input one)
}
