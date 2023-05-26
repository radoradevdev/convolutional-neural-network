#include "ConvolutionalLayer.h"

#include <QTextStream>

using namespace std;

ConvolutionalLayer::ConvolutionalLayer(
        int image_dim[3],
        int kernels[4],
        int padding,
        int stride,
        double bias,
        double eta
    ) {
    srand(time(NULL));

    if (image_dim[0] != kernels[3]) {
        throw std::runtime_error("The depth of the filter must match the depth of the image.");
    }

    // Set image dim and kernels,
    // 3 is the length of the image_dim hyperparameter,
    // 4 of the kernel hyperparameter
    copy(image_dim, image_dim + 3, begin(_image_dim));
    copy(kernels, kernels + 4, begin(_kernels));

    _padding = padding;
    _stride = stride;
    _eta = eta;
    _iteration = 0;
    _filter.init(_kernels, 4);

    int pad_dim[3] = {_image_dim[0], _image_dim[1] + 2 * _padding,
                      _image_dim[2] + 2 * _padding};

    // Update _image_dim with padding
    copy(pad_dim, pad_dim + 3, begin(_image_dim));

    _cache.init(pad_dim, 3);

    for (int k_indx = 0; k_indx < kernels[0]; k_indx++) {
        _bias.push_back(bias);
    }

    // Fill the filter with random values at first
    for (int i = 0; i < _filter.getLength(); i++) {
        _filter[i] = ((double)(arc4random() % 100)) / 1000;
    }
}

void ConvolutionalLayer::_addPadding(const Elements &image, Elements &out) {
    for (int depth = 0; depth < _image_dim[0]; depth++) {
        for (int height = 0; height < _image_dim[1] - 2 * _padding; height++) {
            for (int width = 0; width < _image_dim[2] - 2 * _padding; width++) {

                int output[3] = { depth, height + _padding, width + _padding };
                int input[3] = { depth, height, width };
                out.allocate(image.getValue(input, 3), output, 3);
            }
        }
    }
}

void ConvolutionalLayer::_adjustOutDimensions() {

    int f_height = _kernels[1],  /**< filter height */
        f_width = _kernels[2];   /**< filter width */
    double y, x;

    y = (double)(_image_dim[1] - f_height + 2 * _padding) / _stride + 1;
    x = (double)(_image_dim[2] - f_width + 2 * _padding) / _stride + 1;

    int y_int = (int)(y + 0.5),
        x_int = (int)(x + 0.5);

    _out_dim[0] = _kernels[0]; // The output depth is equal to the number of kernels
    // of the filter
    _out_dim[1] = y_int;
    _out_dim[2] = x_int;
}

void ConvolutionalLayer::fwd(const Elements& image, Elements &out) {

    int f_height = _kernels[1],     /**< filter height */
        f_width = _kernels[2];      /**< filter width */
    int n_kernel = _kernels[0];     /**< filter repetition */

    _adjustOutDimensions();
    int depth = _out_dim[0];

    out.reinit(_out_dim, 3);

    if (_padding != 0)
        _addPadding(image, _cache);
    else
        _cache = image;

    int y_out = 0, x_out = 0;

    for (int kernel_indx = 0; kernel_indx < n_kernel; kernel_indx++) {

        for (int layer = 0; layer < depth; layer++) {
            // each kernel has 3 layers, one for each of the 3
            // layers of the image, the depth.

            y_out = 0, x_out = 0;

            // Calculate the convolution
            for (int y = 0; y < _image_dim[1] - f_height; y += _stride) { // image = ( d x h x w )
                x_out = 0;

                for (int x = 0; x < _image_dim[2] - f_width; x += _stride) {
                    for (int f_y = 0; f_y < f_height; f_y++) {
                        for (int f_x = 0; f_x < f_width; f_x++) {

                            int output[3] = {kernel_indx, y_out, x_out};
                            int cache_pos[3] = {layer, y + f_y, x + f_x};
                            int filtr_pos[4] = {kernel_indx, f_y, f_x, layer};
                            double val =
                                    _cache.getValue(cache_pos, 3) * _filter.getValue(filtr_pos, 4);
                            out.aggregate(val, output, 3);
                        }
                    }
                    x_out++;
                }
                y_out++;
            }
            // add the bias to push the result towards the most probable
            out[kernel_indx] += _bias[kernel_indx];
        }
    }

    // Go through the activation function
    Util::ReLu(out);
}

void ConvolutionalLayer::bp(Elements out, Elements &image) {

    // Neutralize the results from the activation function
    Util::deLeReLu(out);

    int n_kernel = _kernels[0],     /**< filter repetition */
        f_height = _kernels[1],     /**< filter height */
        f_width = _kernels[2],      /**< filter width */
        f_depth = _kernels[3];      /**< filter depth */

    image.reinit(_image_dim, 3);

    // The list of lists of error terms (lowercase deltas)
    Elements d_filters(_kernels[0], _kernels[1], _kernels[2], _kernels[3]);
    vector<double> d_bias;

    int y_out = 0, x_out = 0;

    for (int kernel_indx = 0; kernel_indx < n_kernel; kernel_indx++) {

        y_out = 0, x_out = 0;

        for (int y = 0; y < _image_dim[1] - f_height - 2 * _padding;
             y += _stride) { // image = ( depth x H x W )

            for (int x = 0; x < _image_dim[2] - f_width - 2 * _padding; x += _stride) {

                // loss gradient of the input passed in the convolution operation

                for (int layer = 0; layer < f_depth; layer++) {

                    for (int f_y = 0; f_y < f_height; f_y++) {
                        for (int f_x = 0; f_x < f_width; f_x++) {

                            // position in the filter elements for a specific kernel index,
                            // y-coordinate, x-coordinate, and layer.
                            int filtr_pos[4] = {kernel_indx, f_y, f_x, layer};
                            int output[3] = {layer, y + f_y, x + f_x};

                            // pos in the cache for a specific layer, y-coordinate, and x-coordinate
                            int cache_pos[3] = {layer, y + f_y, x + f_x};
                            // pos in the output for a specific kernel index, y-coordinate, and x-coordinate
                            int kernl_pos[3] = {kernel_indx, y_out, x_out};

                            // value of the partial derivative with respect to the filter weights
                            // represents the contribution of the corresponding filter weight
                            // to the loss gradient during backpropagation.
                            // It helps determine how changes in the filter weights impact
                            // the overall loss of the neural network and is used to
                            // update the filter weights during the gradient descent step
                            double pd_filt = _cache.getValue(cache_pos, 3) *
                                    out.getValue(kernl_pos, 3);

                            if(std::isnan(pd_filt)) {
                                throw std::runtime_error("NaN value!");
                            }

                            // value of the partial derivative with respect to the input
                            // represents the contribution of the corresponding filter weight
                            // to the loss gradient with respect to the input during backpropagation
                            // It helps determine how changes in the input impact the overall
                            // loss of the neural network and is used to
                            // update the input during the backpropagation step
                            double pd_in =
                                    out.getValue(kernl_pos, 3) * _filter.getValue(filtr_pos, 4);

                            // collecting the gradients with respect to the filter weights
                            // during the backpropagation process
                            // These gradients will later be used to update the filter weights
                            // using gradient descent
                            d_filters.aggregate(pd_filt, filtr_pos, 4);

                            // allows for later updates or adjustments to be made based on the
                            // influence of the input on the overall loss
                            image.aggregate(pd_in, output, 3);
                        }
                    }
                }
                x_out++;
            }
            x_out = 0;
            y_out++;
        }
    }

    // loss gradient of the bias
    double k_bias = 0.0;

    for (int kernel_indx = 0; kernel_indx < n_kernel; kernel_indx++) {

        k_bias = 0.0;
        for (int y = 0; y < out.getParam(1); y++) {
            for (int x = 0; x < out.getParam(2); x++) {
                int pos[3] = {kernel_indx, y, x};
                k_bias += out.getValue(pos, 3);
            }
        }
        d_bias.push_back(k_bias);
    }

    // Adjusting the filter weights and bias terms
    _applyGradientDescent(d_filters, d_bias);
}

void ConvolutionalLayer::_applyGradientDescent(Elements &d_filter, vector<double> &d_bias) {

    // Calculating the rate of convergence
    _eta = _eta * exp(((double)-_iteration) / 10000);

    int n_kernel = _kernels[0],     /**< filter repetition */
        f_height = _kernels[1],     /**< filter height */
        f_width = _kernels[2],      /**< filter width */
        f_depth = _kernels[3];      /**< filter depth */

    for (int kernel_indx = 0; kernel_indx < n_kernel; kernel_indx++) {
        for (int y = 0; y < f_height; y++) {
            for (int x = 0; x < f_width; x++) {
                for (int layer = 0; layer < f_depth; layer++) {

                    int index[4] = { kernel_indx, y, x, layer };

                    double delta = -_eta * d_filter.getValue(index, 4);

                    _filter.aggregate(delta, index, 4);
                }
            }
        }
    }

    for (int i = 0; i < (int)_bias.size(); i++) {
        _bias[i] -= _eta * d_bias[i];
    }

    _iteration++;
}

void ConvolutionalLayer::addEpoch(double eta) {

    _eta = eta;
    _iteration = 0;
}
