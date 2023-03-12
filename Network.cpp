#include "Network.h"

void Network::addConvolutionalLayer(vector<int> &image_dim, vector<int> &kernels,
                                int padding, int stride, double bias,
                                double eta) {
    int *dim_ptr = &image_dim[0];
    int *ker_ptr = &kernels[0];

    ConvolutionalLayer layer(dim_ptr, ker_ptr, padding, stride, bias, eta);
    _convs.push_back(layer);
    _layers.push_back(LayerType::Conv);
    _total_layers++;
}

void Network::addFullyConnectedLayer(int input, vector<int> &hidden,
                                 int num_classes, double bias, bool adam,
                                 double eta) {
    vector<int> layers(hidden); // Obtain a copy (since hidden can change outside)
    layers.insert(layers.begin(), input);
    layers.push_back(num_classes);

    FullyConnectedLayer layer(layers, bias, adam, eta);
    _fulls.push_back(layer);
    _num_classes = num_classes;
    _layers.push_back(LayerType::Full);
    _total_layers++;
}

void Network::loadDataset(string data_name) {
    if (data_name == "MNIST") {
        MNIST mnist;

        mnist.getDataset(Train_DS, Train_EV, Test_DS, Test_EV, Valid_DS, Valid_EV);
        _image_shape[2] = Train_DS.get_shape(3);
        _image_shape[1] = Train_DS.get_shape(2);
        _image_shape[0] = Train_DS.get_shape(1);
    }
    // TODO: add more
}

void Network::_forward(Elements &image) {
    Elements img_out;

    for (int i = 0; i < _total_layers; i++) {
        if (_layers[i] == LayerType::Conv) {
            _convs[_conv_index].fwd(image, img_out);
            _conv_index++;
            image = img_out;
        } else if (_layers[i] == LayerType::Pool) {
        } else if (_layers[i] == LayerType::Full) {

            if (_dense_input_shape[0] == 0) {
                for (int i = 0; i < 3; i++) {
                    _dense_input_shape[i] = image.get_shape(i);
                }
            }

            _result = _fulls[0].run(image.get_vector());
        }
    }
}

void Network::_backward(vector<double> &gradient) {
    Elements img_out, img_in;

    for (int i = _total_layers - 1; i >= 0; i--) {
        if (_layers[i] == LayerType::Conv) {
            _conv_index--;
            _convs[_conv_index].bp(img_in, img_out);
            img_in = img_out;
        } else if (_layers[i] == LayerType::Pool) {
        } else if (_layers[i] == LayerType::Full) {
            gradient = _fulls[0].bp(gradient);

            img_in.init(_dense_input_shape, 3);
            img_in.get_vector() = gradient;
        }
    }
}

void Network::_get_image(Elements &image, Elements &dataset, int index) {
    double val;

    image.rebuild(_image_shape, 3);

    for (int d = 0; d < _image_shape[0]; ++d) {
        for (int c = 0; c < _image_shape[1]; ++c) {
            for (int r = 0; r < _image_shape[2]; ++r) {

                int index_ds[4] = {index, d, r, c};
                int index_im[3] = {d, r, c};
                val = dataset.get_value(index_ds, 4);
                image.assign(val, index_im, 3);
            }
        }
    }
}

void Network::_iterate(Elements &dataset, vector<int> &labels,
                   vector<double> &loss_list, vector<double> &acc_list,
                   int preview_period, bool b_training) {

    int label = 0;
    double accuracy = 0, loss = 0, correctAnswer = 0;
    Elements image;

    // stores time in t_start
    time_t t_start;
    time(&t_start);

    int DS_len = dataset.get_shape(0);

    for (int sample = 0; sample < DS_len; sample++) {

        _get_image(image, dataset, sample);
        label = labels[sample];

        // feed the sample into the network
        // he result is stored in _result
        _conv_index = 0;
        _forward(image); //--> _result

        // Error evaluation:
        vector<double> y(_num_classes, 0), error(_num_classes, 0);
        y[label] = 1;

        for (int i = 0; i < _num_classes; i++) {
            error[i] = y[i] - _result[i];
        }

        // update MSE loss function
        double tmp = 0;
        for (int i = 0; i < _num_classes; i++) {
            tmp += pow(error[i], 2);
        }

        loss = tmp / _num_classes;

        loss_list.push_back(loss);

        // reset tmp to recycle it
        tmp = 0;
        for (int i = 0; i < _num_classes; i++) {
            if (_result[i] > _result[tmp]) {
                tmp = i;
            }
        }

        if ((int)tmp == label) {
            correctAnswer++;
        }

        // update accuracy
        accuracy = correctAnswer * 100 / (sample + 1);
        acc_list.push_back(accuracy);

        // adjust the weight
        if (b_training) {
            _backward(error);
        }

        if (sample % preview_period == 0 && sample != 0) {
            double left, total;
            time_t elapsed;
            time(&elapsed);
            total = (double)(elapsed - t_start) / sample * DS_len;
            left = total - (double)(elapsed - t_start);
            printf("\t  Accuracy: %02.2f - Loss: %02.2f - Sample %04d  ||  Label: %d "
                   "- Prediction: %d  ||  Elapsed time: %02.2f - Left time: %02.2f - "
                   "Total time: %02.2f \r",
                   accuracy, loss, sample, label, (int)tmp, (double)elapsed - t_start,
                   left, total);
        }
    }
}

// prew_freq is used to print every preview_period iterations the evolution of
// performances
void Network::train(int epochs, int preview_period) {

    if (_total_layers == 0) {
        cerr << "Error: the network has no layers." << endl;
    } else {
        cout << "\n\n\no Traininig: " << endl;

        for (int epoch = 0; epoch < epochs; epoch++) {
            cout << "\n\to Epoch " << epoch + 1 << endl;
            // Batch Size = 1 => stochastic gradient descent learning algorithm
            _iterate(Train_DS, Train_EV, train_loss, train_acc, preview_period, true);

            cout<<("\nValidating:\n")<<endl;
            //the model evaluation is performed on the validation set after every epoch
            _iterate(Valid_DS, Valid_EV, valid_loss, valid_acc, preview_period, false);
        }
    }
}

// prew_freq is used to print every preview_period iterations the evolution of
// performances
void Network::test(int preview_period) {
    if (_total_layers == 0) {
        cerr << "Error: the network has no layers." << endl;
    } else {
        cout << ("\n\no Testing:") << endl;
        // evaluate the performances on the test dataset
        _iterate(Test_DS, Test_EV, test_loss, test_acc, preview_period, false);
    }
}

void Network::doSanityCheck(int set_size, int epochs) {
    if (_total_layers == 0) {
        cerr << "Error: the network has no layers." << endl;
    } else {

        cout << ("\no Performing sanity check:\n") << endl;

        vector<int> check_EV;
        vector<double> check_loss, check_acc;

        Elements check_DS(set_size, _image_shape[0], _image_shape[1],
                _image_shape[2]);

        for (int sample = 0; sample < set_size; sample++) {
            double val;
            for (int d = 0; d < _image_shape[0]; ++d) {
                for (int c = 0; c < _image_shape[1]; ++c) {
                    for (int r = 0; r < _image_shape[2]; ++r) {
                        int index[4] = {sample, d, r, c};
                        val = Test_DS.get_value(index, 4);
                        check_DS.assign(val, index, 4);
                    }
                }
            }

            check_EV.push_back(Test_EV[sample]);
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            check_loss.clear();
            check_acc.clear();
            printf("\r\to Epoch %d  ||", (epoch + 1));
            _iterate(check_DS, check_EV, check_loss, check_acc, (set_size - 1), true);
        }

        double loss_avg = 0.0;
        for (int i = 0; i < (int)check_loss.size(); i++) {
            loss_avg += check_loss[i] / check_loss.size();
        }

        printf("\n\n\tFinal losses: %02.2f", loss_avg);
    }
}
