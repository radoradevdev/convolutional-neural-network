#include "FullyConnectedLayer.h"

#include <QTextStream>

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

    for (int layer_indx = 0; layer_indx < (int)layers.size(); layer_indx++) {
        _values.push_back(vector<double>(layers[layer_indx], 0.0));
        _d.push_back(vector<double>(layers[layer_indx], 0.0));
        _loss_gradient.push_back(vector<double>(layers[layer_indx], 0.0));
        _network.push_back(vector<Neuron>());

        if (layer_indx > 0) { // network[0] is the input layer, so it doesn't have neurons
            for (int indx = 0; indx < layers[layer_indx]; indx++) {
                _network[layer_indx].push_back(Neuron(layers[layer_indx - 1], bias));
            }
        }
    }
}

void FullyConnectedLayer::setWeights(vector<vector<vector<double>>> weights) {
    for (int weight_y = 0; weight_y < (int)weights.size(); weight_y++) {
        for (int weight_x = 0; weight_x < (int)weights[weight_y].size(); weight_x++) {
            _network[weight_y + 1][weight_x].setWeights(weights[weight_y][weight_x]);
        }
    }
}

void FullyConnectedLayer::printWeights() {
    QTextStream(stdout) << "" << Qt::endl;
    for (int indx = 1; indx < (int)_network.size(); indx++) {
        for (int layer_indx = 0; layer_indx < _sublayers[indx]; layer_indx++) {
            QTextStream(stdout) << "Layer " << indx + 1 << " Neuron " << layer_indx << ": ";
            for (auto &it : _network[indx][layer_indx]._weights)
                QTextStream(stdout) << it << "   ";
            QTextStream(stdout) << "" << Qt::endl;
        }
    }
    QTextStream(stdout) << "" << Qt::endl;
}

vector<double> FullyConnectedLayer::fwd(vector<double> data) {
    _values[0] = data;
    for (int indx = 1; indx < (int)_network.size(); indx++) {
        for (int layer_indx = 0; layer_indx < _sublayers[indx]; layer_indx++) {
            _values[indx][layer_indx] = _network[indx][layer_indx].run(_values[indx - 1]);
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

    past_gradient = FIRST_MOMENT_DECAY * past_gradient + (1 - FIRST_MOMENT_DECAY) * dx;
    double mt = past_gradient / (1 - (double)pow(FIRST_MOMENT_DECAY, t));
    squared_gradient = SECOND_MOMENT_DECAY * squared_gradient + (1 - SECOND_MOMENT_DECAY) * ((double)pow(dx, 2));
    double vt = squared_gradient / (1 - (double)pow(SECOND_MOMENT_DECAY, t));
    double delta = _eta * mt / (sqrt(vt) + EPS);

    return delta;
}

void FullyConnectedLayer::applyGradientDescent() {
    for (int indx = 1; indx < (int)_network.size(); indx++)
        for (int layer_indx = 0; layer_indx < _sublayers[indx]; layer_indx++)
            for (int sublayer_indx = 0; sublayer_indx < _sublayers[indx - 1] + 1; sublayer_indx++) {
                double delta;
                if (sublayer_indx == _sublayers[indx - 1]) {
                    delta = _eta * _d[indx][layer_indx];
                } else {

                    double dw = _d[indx][layer_indx] * _values[indx - 1][sublayer_indx]; // dw = dscore * X

                    if (!_b_adam) {
                        delta = _eta * dw; // learning rate * dw
                    } else {
                        delta = calcAdam(_network[indx][layer_indx]._past_gradient[sublayer_indx], _network[indx][layer_indx]._squared_gradient[sublayer_indx], dw);
                    }
                }
                _network[indx][layer_indx]._weights[sublayer_indx] += delta;
            }
}

vector<double> FullyConnectedLayer::bp(vector<double> error_gradient) {

    _back_iter++; // Updates adam result

    vector<double> outputs = _values.back();

    // Calculate the output error
    for (int out_indx = 0; out_indx < (int)outputs.size(); out_indx++) {
        _d.back()[out_indx] = outputs[out_indx] * (1 - outputs[out_indx]) * (error_gradient[out_indx]);
    }

    // Calculate the error term of each unit on each layer
    for (int indx = ((int)_network.size()) - 2; indx > 0; indx--) {

        for (int layer_size = 0; layer_size < (int)_network[indx].size(); layer_size++) {

            double fwd_error = 0.0;

            for (int sublayer_indx = 0; sublayer_indx < _sublayers[indx + 1]; sublayer_indx++) {
                fwd_error += _network[indx + 1][sublayer_indx]._weights[layer_size] * _d[indx + 1][sublayer_indx];
            }

            _loss_gradient[indx][layer_size] = fwd_error;
            _d[indx][layer_size] = _values[indx][layer_size] * (1 - _values[indx][layer_size]) * fwd_error;
        }
    }

    applyGradientDescent();

    return _loss_gradient[1]; // Result from the first layer (not the input one)
}
