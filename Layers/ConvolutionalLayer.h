#ifndef CONVOLUTIONALLAYER_H
#define CONVOLUTIONALLAYER_H

#include <algorithm>
#include <ctime>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <stdlib.h>
#include <vector>

#include "Elements.h"
#include "Util.h"

using namespace std;

//! ConvolutionalLayer class
/*!
  The class describes the operations of the convolutional layers.
*/
class ConvolutionalLayer {
public:
    //! Constructor for initializing the Convolutional Layer
    /*!
      \param image_dim  image dimensions
      \param kernels    kernel of the convolution.
      \param padding    padding on all sides, for the kernel to operate
      \param stride     how many pixels every convolution to shift
      \param bias       constant to offset the weights
      \param eta        learning rate parameter, size of step towards loss=0
    */
    ConvolutionalLayer(
            int image_dim[3],
            int kernels[4],
            int padding = 1,
            int stride = 1,
            double bias = 0.1,
            double eta = 0.01
            );

    //! Adds a new epoch
    /*!
      \param eta    learning rate parameter, size of step towards loss=0
    */
    void addEpoch(double eta);

    //! Forward propagation in the convolutional layer
    /*!
      \param image  original image
      \param out    result
    */
    void fwd(const Elements &image, Elements &out);

    //! Backwards propagation in the convolutional layer
    /*!
      \param out    result
      \param image  original image
    */
    void bp(Elements out, Elements &image);
private:
    int _image_dim[3] = {1, 16, 16};    /*!< Default image specification */
    int _kernels[4] = {2, 3, 3, 1};     /*!< Default kernel spcification */
    int _out_dim[3] = {2, 13, 13};      /*!< depth, height, width */

    int _padding = 1;   /*!< Padding on all sides, for the kernel to operate */
    int _stride = 2;    /*!< How many pixels every convolution to shift */
    int _iteration = 0; /*!< TODO Iterations over the Convolutional layer, for 1 epoch */
    double _eta = 0.1;  /*!< learning rate parameter, size of step towards loss=0 */

    vector<double> _bias; /*!< TODO Constant to offset the weights */

    Elements _cache;    /*!< the cached forward elements */
    Elements _filter;   /*!< firstly, it holds randomized values and
                            then it holds the result of the gradient descent */

    //! Enlarge the image and _image_dim is changed accordingly
    /*!
     \param image   original image
      \param out    result
    */
    void _addPadding(const Elements &image, Elements &out);

    //! Applies the gradient descent
    /*!
      \param d_filter   TODO
      \param d_bias     TODO
    */
    void _applyGradientDescent(Elements &d_filter, vector<double> &d_bias);

    //! Adjusts the output dimensions
    void _adjustOutDimensions();
};

#endif
