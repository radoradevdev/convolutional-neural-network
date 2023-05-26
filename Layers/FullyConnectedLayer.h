#ifndef FULLYCONNECTEDLAYER_H
#define FULLYCONNECTEDLAYER_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <numeric>
#include <random>
#include <time.h>
#include <vector>

#include "Util.h"
#include "Neuron.h"

//! Hyperparameters for the Adam function
#define FIRST_MOMENT_DECAY 0.9
#define SECOND_MOMENT_DECAY 0.999
#define EPS 1e-7

using namespace std;

//! Fully connected layer class
/*!
  The class describes the oprations of a multi-layered perceptron.
  And it will compute the class scores, after the convolutional layer.
*/
class FullyConnectedLayer {
public:
    //! Constructor for initializing the FullyConnected Layer
    /*!
      \param layers  the layers, [0] = number of perceptrons
      \param bias    constant to offset the weights
      \param adam    padding on all sides, for the kernel to operate
      \param eta     learning rate parameter, size of step towards loss=0
    */
    FullyConnectedLayer(
            vector<int> layers,
            double bias = 1.0,
            bool adam = true,
            double eta = 0.5
            );

    //! Set the weights
    /*!
      \param weights  initial weights, without the input layer
    */
    void setWeights(vector<vector<vector<double>>> weights);

    //! Print the weights
    void printWeights();

    //! Forward propagation
    /*!
      \param data   initial weights
    */
    vector<double> fwd(vector<double> data);

    //! Backwards propagation
    /*!
      \param error_gradient gradient errors
    */
    vector<double> bp(vector<double> error_gradient);

    //! Applies the gradient descent, calculates the deltas and updates the weights
    void applyGradientDescent();

    //! Calculates Adam(Adaptive Moment Estimation)
    /*!
      \param past_gradient      past gradient
      \param squared_gradient   squared gradient
      \param derivative         derivative
    */
    double calcAdam(
            double &past_gradient,
            double &squared_gradient,
            double derivative
            );

private:
    vector<int> _sublayers;     /*!< sublayers in the fully connected layer */
    double _bias;               /*!< constant to offset the weights */
    double _eta;                /*!< learning rate parameter, size of step towards loss=0 */

    int _back_iter;             /*!< backpropagation iterations */
    bool _b_adam;               /*!< apply adam or not? */

    vector<vector<Neuron>> _network;        /*!< network of perceptrons */
    vector<vector<double>> _values;         /*!< values obtained in forward prop */
    vector<vector<double>> _d;              /*!< values obtained in backward prop */
    vector<vector<double>> _loss_gradient;  /*!< errors */
};

#endif
