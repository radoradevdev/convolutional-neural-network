#include "Network.h"

#include <QtCharts/QChart>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>

#include <QLabel>

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

    // Initialise the convolutional layer with the hyperparameters
    ConvolutionalLayer layer(dim_ptr, ker_ptr, padding, stride, bias, eta);
    _convs.push_back(layer);
    _layers.push_back(LayerType::Conv);
    _total_lrs++;
}

void Network::addPoolingLayer(
        vector<int> &image_dim,
        PoolingOperation mode,
        int size,
        int stride
        ) {
    int *dim_ptr = &image_dim[0];

    PoolingLayer layer(dim_ptr, mode, size, stride);
    _pools.push_back(layer);
    _layers.push_back(LayerType::Pool);
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
    vector<int> sublayers(hidden); // form sublayers, copy vector from hidden
    sublayers.insert(sublayers.begin(), input); // push_front, size of the layer
    sublayers.push_back(num_classes); // output layer
    // At this point layers = { input, hidden, num_classes }

    // Initialise the fully connected layer with the hyperparameters
    FullyConnectedLayer layer(sublayers, bias, adam, eta);
    _fulls.push_back(layer);
    _num_clss = num_classes;
    _layers.push_back(LayerType::Full);
    _total_lrs++;
}

void Network::loadDataset(DatasetType dataset) {
    if (dataset == DatasetType::MNIST) {
        class MNIST mnist;

        mnist.getDataset(Train_DS, Train_EV, Valid_DS, Valid_EV, Test_DS, Test_EV);
    } else if(dataset == DatasetType::CARTEDUCIEL) {
        class CarteDuCiel cdc;

        cdc.getDataset(Train_DS, Train_EV, Test_DS, Test_EV);
    }

    _image_shape[2] = Train_DS.getParam(3); // width
    _image_shape[1] = Train_DS.getParam(2); // height
    _image_shape[0] = Train_DS.getParam(1); // depth
}

void Network::_forward(Elements &image, bool b_plot) {
    Elements img_out; /**< The modified image */

    // Go through all layers
    for (int layer_indx = 0; layer_indx < _total_lrs; layer_indx++) {
        if(b_plot) {
            plotList.append(Util::elementsToQImage(image));
        }
        if (_layers[layer_indx] == LayerType::Conv) {
            // Forward convolution
            _convs[_conv_indx].fwd(image, img_out); // debug this line to verify the hyperparams

            // move the conv_indx forward,
            // because there can be more than one conv layers
            _conv_indx++;

            image = img_out; // debug this line to verify the hyperparams
        } else if (_layers[layer_indx] == LayerType::Pool) {
            // Forward Pooling
            _pools[_pool_indx].fwd(image, img_out); // debug this line to verify the hyperparams

            // move the pool_indx forward,
            // because there can be more than one pool layers
            _pool_indx++;

            image = img_out; // debug this line to verify the hyperparams
        } else if (_layers[layer_indx] == LayerType::Full) {

            if (_dense_input_shape[0] == 0) {
                _dense_input_shape[0] = image.getParam(0); // layers
                _dense_input_shape[1] = image.getParam(1); // height
                _dense_input_shape[2] = image.getParam(2); // width
            }
            // Forward propagation
            // In this implementation, there can be only one FC layer, thats why the 0
            _results = _fulls[0].fwd(image.getData());
        }
    }
}

void Network::_backward(vector<double> &gradient) {
    Elements img_out, img_in;

    for (int layer_indx = _total_lrs - 1; layer_indx >= 0; layer_indx--) {
        if (_layers[layer_indx] == LayerType::Conv) {
            _conv_indx--;

            // Backwards convolution
            _convs[_conv_indx].bp(img_in, img_out);
            img_in = img_out;
        } else if (_layers[layer_indx] == LayerType::Pool) {
            _pool_indx--;

            // Backwards Pooling
            _pools[_pool_indx].bp(img_in, img_out);
            img_in = img_out;
        } else if (_layers[layer_indx] == LayerType::Full) {
            // Backwards propagation
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
                image.allocate(val, index_im, 3);
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
        bool b_training,
        bool b_plot
        ) {

    int expected_value = 0;
    double accuracy = 0, loss = 0, count_recognized = 0;
    Elements image;

    time_t t_start;
    time(&t_start);

    int dataset_size = dataset.getParam(0);

    for (int sample_indx = 0; sample_indx < dataset_size; sample_indx++) {

        // Load the image
        _getImage(image, dataset, sample_indx);
        // Load the expected value class
        expected_value = expected_values[sample_indx];

        // the result of the forward propagation is stored in _results
        _conv_indx = 0;
        _forward(image, b_plot);

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
        double res = 0.0;
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

            QTextStream out(stdout);
            out << "\r\t  Accuracy: " << accuracy << " - "
                << "Loss: " << loss << " - "
                << "Sample " << sample_indx << " || "
                << "Label: " << expected_value << " - "
                << "Prediction: " << (int)res << " || "
                << "Elapsed time: " << (double)elapsed - t_start << " - "
                << "Left time: " << left << " - "
                << "Total time: " << total << Qt::endl;

            out.flush();
        }

        b_plot = false;
    }
}


void Network::train(int epochs, int preview_interval, bool doValidate) {

    QTextStream(stdout) <<"\n\n> Traininig: " << Qt::endl;

    for (int epoch = 0; epoch < epochs; epoch++) {
        QTextStream(stdout) << "\n\t> Epoch " << epoch + 1 << Qt::endl;
        _iterate(Train_DS, Train_EV, train_loss, train_acc, preview_interval, true, false);

        if(doValidate) {
            QTextStream(stdout) << ("\nValidating:\n") << Qt::endl;
            // the model evaluation is performed on the validation set after every epoch
            _iterate(Valid_DS, Valid_EV, valid_loss, valid_acc, preview_interval, false);
        }
    }
}

void Network::test(int preview_interval) {
    QTextStream(stdout) <<("\n\n> Testing:") << Qt::endl;
    _iterate(Test_DS, Test_EV, test_loss, test_acc, preview_interval, false);
}

void Network::checkConfiguration(int set_size, int epochs) {
    QTextStream(stdout) << ("\n> Performing check:\n") << Qt::endl;

    vector<int> Check_EV;
    vector<double> check_loss, check_acc;

    Elements check_DS(set_size, _image_shape[0], _image_shape[1],
            _image_shape[2]);

    // Loading the temporary Check_DS and Check_EV
    for (int sample = 0; sample < set_size; sample++) {
        double val;
        for (int d = 0; d < _image_shape[0]; ++d) {
            for (int c = 0; c < _image_shape[1]; ++c) {
                for (int r = 0; r < _image_shape[2]; ++r) {
                    int index[4] = {sample, d, r, c};
                    val = Test_DS.getValue(index, 4);
                    check_DS.allocate(val, index, 4);
                }
            }
        }

        Check_EV.push_back(Test_EV[sample]);
    }

    // test with Check_DS and Check_EV
    for (int epoch = 0; epoch < epochs; epoch++) {
        check_loss.clear();
        check_acc.clear();
        QTextStream out(stdout);
        out << "\t> Epoch " << (epoch + 1) << Qt::endl;
        out.flush();
        _iterate(check_DS, Check_EV, check_loss, check_acc, (set_size - 1), true);
    }

    double loss_avg = 0.0;
    for (int i = 0; i < (int)check_loss.size(); i++) {
        loss_avg += check_loss[i] / check_loss.size();
    }

    // The losses should be closed to 0
    QTextStream(stdout) << "\n\n\tFinal losses: " <<  loss_avg;
}

void Network::plotResults(bool doValidate) {
    // Create chart and set title
    QChart *chart = new QChart();
    chart->setTitle("Accuracy % /Loss");

    // Create series and add data
    QLineSeries *trainAccSeries = new QLineSeries();
    QLineSeries *trainLossSeries = new QLineSeries();
    QLineSeries *validAccSeries = new QLineSeries();
    QLineSeries *validLossSeries = new QLineSeries();

    trainAccSeries->setName("Train Accuracy");
    trainLossSeries->setName("Train Loss");
    if(doValidate) {
        validAccSeries->setName("Validation Accuracy");
        validLossSeries->setName("Validation Loss");
    }

    int lenTrain = train_acc.size();
    int deltaTrain = 1;
    for (int i = 0; i < lenTrain; i += deltaTrain) {
        trainAccSeries->append(i, train_acc[i]);
        trainLossSeries->append(i, train_loss[i]);
    }

    if(doValidate) {
        int lenValid = 100; // valid_acc.size();
        int deltaValid = 1;
        for (int i = 0; i < lenValid; i += deltaValid) {
            if(isnan(valid_loss[i])) {
                valid_loss[i] = 1; // TODO: remove and figure out why there are nans
            }
            validAccSeries->append(i, valid_acc[i]);
            validLossSeries->append(i, valid_loss[i]);
        }
    }

    // Add series to chart
    chart->addSeries(trainAccSeries);
    chart->addSeries(trainLossSeries);
    if(doValidate) {
        chart->addSeries(validAccSeries);
        chart->addSeries(validLossSeries);
    }

    // Set axis titles
    QValueAxis *xAxis = new QValueAxis;
    xAxis->setTitleText("Iterations");
    chart->addAxis(xAxis, Qt::AlignBottom);

    QValueAxis *yAxisA = new QValueAxis;
    yAxisA->setTitleText("Accuracy %");
    chart->addAxis(yAxisA, Qt::AlignLeft);

    QValueAxis *yAxisL = new QValueAxis;
    yAxisL->setTitleText("Loss");
    chart->addAxis(yAxisL, Qt::AlignRight);

    // Attach series to axes
    trainAccSeries->attachAxis(xAxis);
    trainAccSeries->attachAxis(yAxisA);
    trainLossSeries->attachAxis(xAxis);
    trainLossSeries->attachAxis(yAxisL);
    if(doValidate) {
        validAccSeries->attachAxis(xAxis);
        validAccSeries->attachAxis(yAxisA);
        validLossSeries->attachAxis(xAxis);
        validLossSeries->attachAxis(yAxisL);
    }

    // Create chart view and add chart to it
    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // Show chart view
    chartView->show();
}

void Network::plotFilteredImages() {
    int rows = std::ceil(std::sqrt(plotList.size()));
    int cols = rows;

    // create a layout for the labels
//    QGridLayout* layout = new QGridLayout();

//    QImage heatmapImage = plotList[0];
    QImage heatmapImage =  Util::grayscaleToHeatmap(plotList[0]);
    QImage resizedImage = heatmapImage.scaled(200, 200);


    // add the first image to the top-left cell
    QLabel* inputLabel = new QLabel();
    inputLabel->setPixmap(QPixmap::fromImage(resizedImage));
    inputLabel->show();
//    inputLabel->setAlignment(Qt::AlignCenter);
//    inputLabel->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//    layout->addWidget(inputLabel, 0, 0);

    // add the rest of the images
//    for (int i = 1; i < plotList.size(); i++) {
//        QLabel* label = new QLabel();
//        label->setPixmap(QPixmap::fromImage(plotList[i]));
//        label->setAlignment(Qt::AlignCenter);
//        label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
//        layout->addWidget(label, i / cols, i % cols);
//    }

    // create the preview widget
//    QWidget* previewWidget = new QWidget();
//    previewWidget->setLayout(layout);
//    previewWidget->setWindowTitle("Preview");
//    previewWidget->show();
}
