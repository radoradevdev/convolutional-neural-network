#ifndef NETWORK_H
#define NETWORK_H

#include <ctime>
#include <math.h>
#include <string.h>
#include <vector>

#include "Datasets/MNIST.h"
#include "Layers/ConvolutionalLayer.h"
#include "Layers/PoolingLayer.h"
#include "Layers/FullyConnectedLayer.h"
#include "Elements.h"

using namespace std;

enum LayerType { Conv, Pool, Full };

class Network {
public:
    Network(){};

    // Adds a Convolutional Layer
    void addConvolutionalLayer(vector<int> &image_dim, vector<int> &kernels,
                               int padding = 1, int stride = 1, double bias = 0.1,
                               double eta = 0.01);

    // Adds a Pooling Layer
    void addPoolingLayer(int image_dim[3], char mode = 'a', int size = 2,
    int stride = 2, int padding = 0);

    // Adds a Fully Connected Layer
    void addFullyConnectedLayer(int input, vector<int> &hidden,
                                int num_classes = 10, double bias = 1.0,
                                bool adam = true, double eta = 0.01);

    // Load the dataset
    void loadDataset(string dataset_name);

    // Train the network, with training data
    void train(int epochs = 1, int preview_period = 1);

    // Test the network, with test data
    void test(int preview_period = 1);

    // Sanitaze the network
    void doSanityCheck(int set_size = 50, int epochs = 200);

    // Plot the results
    void plotResults();

private:
    // All layers in the CNN
    vector<LayerType> _layers;

    //
    vector<double> _result;

    // All Convolutional Layers
    vector<ConvolutionalLayer> _convs;

    // All Pooling Layers
    vector<PoolingLayer> _pools;

    // All FullyConnected Layers
    vector<FullyConnectedLayer> _fulls;

    int _conv_index = 0, _pool_index = 0, _total_layers = 0, _num_classes = 0;
    int _dense_input_shape[3] = {0, 0, 0};
    int _image_shape[3] = {0, 0, 0};

    // Data
    Elements Train_DS, Test_DS, Valid_DS;

    // Expected Results (Labels)
    vector<int> Train_EV, Test_EV, Valid_EV;

    vector<double> train_acc, valid_acc, test_acc;
    vector<double> train_loss, valid_loss, test_loss;

    void _forward(Elements &image);
    void _backward(vector<double> &gradient);
    void _iterate(Elements &dataset, vector<int> &labels, vector<double> &loss_list,
                  vector<double> &acc_list, int preview_period,
                  bool b_training = true);
    void _get_image(Elements &image, Elements &dataset, int index);
};

#endif
