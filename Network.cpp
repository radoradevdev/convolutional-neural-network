#include "Network.h"

void Network::addConvolutionalLayer(
        vector<int> &image_dim,
        vector<int> &kernels,
        int padding,
        int stride,
        double bias,
        double eta
        ) {
    int *dim_ptr = &image_dim[0];
    int *ker_ptr = &kernels[0];

    ConvolutionalLayer layer(dim_ptr, ker_ptr, padding, stride, bias, eta);
    _convs.push_back(layer);
    _layers.push_back(LayerType::Conv);
    _total_lrs++;
}

void Network::addFullyConnectedLayer(
        int input,
        vector<int> &hidden,
        int num_classes,
        double bias,
        bool adam,
        double eta
        ) {
    vector<int> layers(hidden); // form sublayers, copy vector from hidden
    layers.insert(layers.begin(), input); // push_front, size of the layer
    layers.push_back(num_classes); // output layer
    // At this point layers = { input, hidden, num_classes }

    FullyConnectedLayer layer(layers, bias, adam, eta);
    _fulls.push_back(layer);
    _num_clss = num_classes;
    _layers.push_back(LayerType::Full);
    _total_lrs++;
}

void Network::loadDataset(DatasetType dataset) {
    if (dataset == DatasetType::MNIST) {
        class MNIST mnist;

        mnist.getDataset(Train_DS, Train_EV, Valid_DS, Valid_EV, Test_DS, Test_EV);

        _image_shape[2] = Train_DS.getParam(3); // width
        _image_shape[1] = Train_DS.getParam(2); // height
        _image_shape[0] = Train_DS.getParam(1); // depth
    }
    // TODO: add more
}

void Network::_forward(Elements &image) {
    Elements img_out; /**< The modified image */

    // Go through all layers
    for (int layer_indx = 0; layer_indx < _total_lrs; layer_indx++) {
        if (_layers[layer_indx] == LayerType::Conv) {
            _convs[_conv_indx].fwd(image, img_out);
            _conv_indx++; // move the conv_indx forward, because there can be more than one conv layers
            image = img_out;
        } else if (_layers[layer_indx] == LayerType::Pool) {
        } else if (_layers[layer_indx] == LayerType::Full) {

            if (_dense_input_shape[0] == 0) {
                _dense_input_shape[0] = image.getParam(0);
                _dense_input_shape[1] = image.getParam(1);
                _dense_input_shape[2] = image.getParam(2);
            }

            _results = _fulls[0].fwd(image.getData());
        }
    }
}

void Network::_backward(vector<double> &gradient) {
    Elements img_out, img_in;

    for (int layer_indx = _total_lrs - 1; layer_indx >= 0; layer_indx--) {
        if (_layers[layer_indx] == LayerType::Conv) {
            _conv_indx--;
            _convs[_conv_indx].bp(img_in, img_out);
            img_in = img_out;
        } else if (_layers[layer_indx] == LayerType::Pool) {
        } else if (_layers[layer_indx] == LayerType::Full) {
            gradient = _fulls[0].bp(gradient);

            img_in.init(_dense_input_shape, 3);
            img_in.getData() = gradient;
        }
    }
}

void Network::_getImage(Elements &image, Elements &dataset, int index) {
    double val;

    image.reinit(_image_shape, 3);

    for (int depth = 0; depth < _image_shape[0]; ++depth) {
        for (int width = 0; width < _image_shape[1]; ++width) {
            for (int height = 0; height < _image_shape[2]; ++height) {

                int index_ds[4] = {index, depth, height, width};
                int index_im[3] = {depth, height, width};
                val = dataset.getValue(index_ds, 4);
                image.assign(val, index_im, 3);
            }
        }
    }
}

void Network::_iterate(
        Elements &dataset,
        vector<int> &expected_values,
        vector<double> &losses,
        vector<double> &accuracies,
        int preview_interval,
        bool b_training
        ) {

    int expected_value = 0;
    double accuracy = 0, loss = 0, count_recognized = 0;
    Elements image;

    time_t t_start;
    time(&t_start);

    int dataset_size = dataset.getParam(0);

    for (int sample_indx = 0; sample_indx < dataset_size; sample_indx++) {

        _getImage(image, dataset, sample_indx);
        expected_value = expected_values[sample_indx];

        // the result is stored in _results
        _conv_indx = 0;
        _forward(image);

        // Error evaluation:
        vector<double> y(_num_clss, 0), error(_num_clss, 0);
        y[expected_value] = 1;

        for (int class_indx = 0; class_indx < _num_clss; class_indx++) {
            error[class_indx] = y[class_indx] - _results[class_indx];
        }

        // update MSE(Mean Squared Error) loss function
        // It is used to measure the difference between the predicted output of a model
        // and the actual target output.
        // The MSE loss function is defined as the average of the squared differences
        // between the predicted values and the actual values
        double res = 0;
        for (int class_indx = 0; class_indx < _num_clss; class_indx++) {
            res += pow(error[class_indx], 2);
        }

        loss = res / _num_clss;

        losses.push_back(loss);

        res = 0.0;
        for (int class_indx = 0; class_indx < _num_clss; class_indx++) {
            if (_results[class_indx] > _results[res]) {
                res = class_indx;
            }
        }

        if ((int)res == expected_value) {
            count_recognized++;
        }

        // Update accuracy
        accuracy = count_recognized * 100 / (sample_indx + 1);
        accuracies.push_back(accuracy);

        // Adjust the weights
        if (b_training) {
            _backward(error);
        }

        if (sample_indx % preview_interval == 0 && sample_indx != 0) {
            double left, total;
            time_t elapsed;
            time(&elapsed);
            total = (double)(elapsed - t_start) / sample_indx * dataset_size;
            left = total - (double)(elapsed - t_start);
            printf("\t  Accuracy: %02.2f - Loss: %02.2f - Sample %04d  ||  Label: %d "
                   "- Prediction: %d  ||  Elapsed time: %02.2f - Left time: %02.2f - "
                   "Total time: %02.2f \r",
                   accuracy, loss, sample_indx, expected_value, (int)res, (double)elapsed - t_start,
                   left, total);
        }
    }
}


void Network::train(int epochs, int preview_interval) {

    /* Instead of computing the gradient over the entire dataset (batch gradient descent),
    the SGD algorithm computes the gradient over a
    randomly selected subset of the data (mini-batch).
    This approach makes the computation of the gradient more efficient
    and reduces the variance of the gradient estimate.

    The basic steps of the SGD algorithm are:

    Initialize the model parameters with small random values.

    Repeat until convergence:

    a. Select a mini-batch of samples from the training dataset.

    b. Compute the gradient of the loss function with respect to the parameters
    for the selected mini-batch.

    c. Update the parameters by taking a small step in the direction
    of the negative gradient.

    d. Evaluate the model on the validation dataset to monitor its performance
    and prevent overfitting. */

    cout << "\n\no Traininig: " << endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        cout << "\n\to Epoch " << epoch + 1 << endl;
        _iterate(Train_DS, Train_EV, train_loss, train_acc, preview_interval, true);

        cout<<("\nValidating:\n")<<endl;
        // the model evaluation is performed on the validation set after every epoch
        _iterate(Valid_DS, Valid_EV, valid_loss, valid_acc, preview_interval, false);
    }
}

void Network::test(int preview_interval) {
    cout << ("\n\no Testing:") << endl;
    _iterate(Test_DS, Test_EV, test_loss, test_acc, preview_interval, false);
}

void Network::checkConfiguration(int set_size, int epochs) {
    cout << ("\no Performing check:\n") << endl;

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
                    val = Test_DS.getValue(index, 4);
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
