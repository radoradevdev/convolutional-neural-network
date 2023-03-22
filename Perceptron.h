#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <numeric>
#include <vector>

#include "Util.h"

using namespace std;

//! Perceptron class
/*!
  The class describes the oprations of a single perceptron in a layer
*/
class Perceptron {
public:
    //! Constructor for initializing the Perceptron
    /*!
      \param inputs  count of inputs
      \param bias    constant to offset the weights
    */
    Perceptron(int inputs, double bias = 1.0);

    //! Constructor for initializing the Perceptron
    /*!
      \param x  values
    */
    double run(vector<double> x);

    //! Set weights
    /*!
      \param init_weights   inital weights
    */
    void setWeights(vector<double> init_weights);

    //! Sigmoid activation function
    /*!
      \param x  value
    */
    double sigmoid(double x);

    vector<double> weights,             /*!< weights */
                    past_gradient,      /*!< past gradients */
                    squared_gradient;   /*!< squared gradients */
private:
    double _bias;                       /*!< constant to offset the weights */
};

#endif // PERCEPTRON_H
