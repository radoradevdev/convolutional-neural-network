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

//! LayerType Enum
/*! It consists of three possible layer types */
enum LayerType {
    Conv,   /*!< Convolutional Layer */
    Pool,   /*!< Pooling Layer */
    Full    /*!< Fully Connected Layer */
};

//! Dataset Enum
enum DatasetType {
    MNIST,   /*!< Handwritten digits */
};

//!  The Main Engine of the program
/*!
  The class containing the description of the neural network
  (layers, datasets, results, analysis and the main actions for training and testing)
*/
class Network {
public:
    //! Adds a Convolutional Layer
    /*!
      \param image_dim  image dimensions
      \param kernels    kernel of the convolution.
      \param padding    padding on all sides, for the kernel to operate
      \param stride     how many pixels every convolution to shift
      \param bias       constant to offset the weights
      \param eta        TODO
    */
    void addConvolutionalLayer(
            vector<int> &image_dim,
            vector<int> &kernels,

            int padding = 1,
            int stride = 1,
            double bias = 0.1,
            double eta = 0.01
            );

    //! TODO Adds a Pooling Layer
    void addPoolingLayer(
            int image_dim[3],
            char mode = 'a',
            int size = 2,
            int stride = 2,
            int padding = 0
            );

    //! Adds a Fully Connected Layer
    /*!
      \param input          size of input layer
      \param hidden         size of hidden layer
      \param num_classes    number of classes to classified
      \param bias           constant to offset the weights
      \param adam           apply adam function?
      \param eta            TODO
    */
    void addFullyConnectedLayer(
            int input,
            vector<int> &hidden,
            int num_classes = 10,
            double bias = 1.0,
            bool adam = true,
            double eta = 0.01
            );

    //! Loads a dataset.
    /*!
      \param dataset_name   name of the dataset
    */
    void loadDataset(DatasetType dataset);

    //! Trains the network, with training data. Updates the weights after each epoch.
    /*!
      \param epochs         how many times to run the training set through the network
      \param preview_interval preview period of the output display
    */
    void train(int epochs = 1, int preview_interval = 1);

    //! Tests the network, with test data, against the weights gained from training.
    /*!
      \param preview_interval preview period of the output display
    */
    void test(int preview_interval = 1);

    //! Checks if the configuration is correct, the loss should go to 0 if so.
    //! Loads set_size number of images from the dataset.
    //! Feeds the data through the network with the current configuration.
    //! Mainly used to quickly check if everything is okay with the configuration.
    //! It shouldn't be used everytime, as it eats time.
    /*!
      \param set_size   how many test images to use
      \param epochs     how many times to run the check set through the network
    */
    void checkConfiguration(int set_size = 50, int epochs = 200);

    //! TODO Plot the results
    void plotResults();

private:
    vector<LayerType> _layers; /*!< All layers in the CNN */

    vector<double> _results; /*!< Results */

    vector<ConvolutionalLayer> _convs; /*!< All Convolutional Layers */

    vector<PoolingLayer> _pools; /*!< All Pooling Layers */

    vector<FullyConnectedLayer> _fulls; /*!< All FullyConnected Layers, TODO to not be a vector */

    int _conv_indx = 0,    /*!< Index of the current convolutional layer */
        _pool_indx = 0,    /*!< Index of the current pool layer */
        _total_lrs = 0,  /*!< Total number of layers */
        _num_clss = 0;   /*!< Number of classes, in the MNIST example there are 10 */

    //! TODO
    int _dense_input_shape[3] = {0, 0, 0};

    //! depth(of activation/feature maps), width, height
    int _image_shape[3] = {0, 0, 0};

    Elements Train_DS,   /*!< Dataset for initial training, (50K images, MNIST) */
             Valid_DS,   /*!< Dataset for validation after each epoch, (10K images, MNIST)  */
             Test_DS;    /*!< Dataset for final testing, (10K images, MNIST) */

    vector<int> Train_EV, /*!< Expected values for Train_DS (Labels) */
                Valid_EV, /*!< Expected values for Valid_DS (Labels) */
                Test_EV;  /*!< Expected values for Test_DS (Labels) */

    vector<double> train_acc, /*!< Training accuracy */
                   valid_acc, /*!< Validition accuracy */
                   test_acc;  /*!< Testing accuracy */

    vector<double> train_loss, /*!< Training losses */
                   valid_loss, /*!< Training losses */
                   test_loss;  /*!< Training losses */

    //! Performs forward propagation
    /*!
      \param image current image
    */
    void _forward(Elements &image);

    //! Performs backwards propagation
    /*!
      \param gradient TODO backwards gradient data
    */
    void _backward(vector<double> &gradient);

    //! Iterates over a dataset to update the loss and the accuracy
    /*!
      \param dataset            dataset to be used
      \param expected_values    expected results
      \param loss_list          losses
      \param acc_list           accuracy
      \param preview_interval   preview period of the output display
      \param b_training         to update the weights or not?
    */
    void _iterate(
            Elements &dataset,
            vector<int> &expected_values,
            vector<double> &loss_list,
            vector<double> &acc_list,
            int preview_interval,
            bool b_training = true
            );

    //! Gets specific image from the dataset by index
    /*!
      \param image      image to be filled
      \param dataset    dataset to get from
      \param index      index of the image
    */
    void _getImage(
            Elements &image,
            Elements &dataset,
            int index
            );
};

#endif
