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
        QTextStream(stderr) << "The depth of the filter must match the depth of the image." << Qt::endl;
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

    // TODO delete
//    double inputs = (kernels[0] * kernels[1] * kernels[2] * kernels[3]);

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
                out.assign(image.getValue(input, 3), output, 3);
            }
        }
    }

    // TODO delete
    // for(int i=0; i<3; i++) _image_dim[i]=out_pad.get_shape(i);
}

void ConvolutionalLayer::_adjustOutDimensions() {

    int f_height = _kernels[1],  /**< filter height */
        f_width = _kernels[2];  /**< filter width */
    double y_doub, x_doub;

    y_doub = (double)(_image_dim[1] - f_height + 2 * _padding) / _stride + 1;
    x_doub = (double)(_image_dim[2] - f_width + 2 * _padding) / _stride + 1;

    int y_int = (int)(y_doub + 0.5), x_int = (int)(x_doub + 0.5);

    // TODO delete
    // if( y_doub!=y_int || x_doub!=x_int) cerr<<"\nWarning: padding and stride
    // combination is not integer."<<endl;

    _out_dim[0] = _kernels[0]; // The output depth is equal to the number of kernels
    // of the filter
    _out_dim[1] = y_int;
    _out_dim[2] = x_int;

    // TODO delete
    // printf("Out dimensions as calculated: %d %d %d\n",_out_dim[0], _out_dim[1],
    // _out_dim[2] );
}

void ConvolutionalLayer::fwd(const Elements& image, Elements &out) {

    // TODO delete
    // Produces a volume of size D2xH2xW2 where:
    //  #W2=(W1−F+2P)/S+1
    //  #H2=(H1−F+2P)/S+1
    //  #D2= kernels number

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
            // each kernel has n (3) layers, one for each of the n (3)
            // layers of the image, the depth.

            y_out = 0, x_out = 0;

            // Calculate the convolution
            for (int y = 0; y < _image_dim[1] - f_height; y += _stride) { // image = ( d x h x w )
                x_out = 0;

                for (int x = 0; x < _image_dim[2] - f_width; x += _stride) {

                    for (int f_y_it = 0; f_y_it < f_height; f_y_it++) {
                        for (int f_x_it = 0; f_x_it < f_width; f_x_it++) {

                            int arr_out[3] = {kernel_indx, y_out, x_out};
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            int in_filt[4] = {kernel_indx, f_y_it, f_x_it, layer};
                            double val =
                                    _cache.getValue(in_cache, 3) * _filter.getValue(in_filt, 4);
                            out.add(val, arr_out, 3);
                        }
                    }
                    x_out++;
                }
                y_out++;
            }
            out[kernel_indx] += _bias[kernel_indx];
        }
    }

    // Go through the activation function
    Util::ReLu(out);
}

void ConvolutionalLayer::bp(Elements out, Elements &image) {

    // Neutralize the results from the activation function
    Util::deLeReLu(out);

    // TODO delete
    // image (input or convolution result) - ( depth, out_H, out_W )
    // _specs = (n_kern x H x W x depth) The filters are in _filters

    int n_kernel = _kernels[0],     /**< filter repetition */
        f_height = _kernels[1],     /**< filter height */
        f_width = _kernels[2],      /**< filter width */
        f_depth = _kernels[3];      /**< filter depth */

    // TODO delete
    // d_input = np.zeros( (  self.padded_dim[0], self.padded_dim[1],
    // self.padded_dim[2]) )

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

                    for (int f_y_it = 0; f_y_it < f_height; f_y_it++) {
                        for (int f_x_it = 0; f_x_it < f_width; f_x_it++) {

                            int filt[4] = {kernel_indx, f_y_it, f_x_it, layer};
                            int out_in[3] = {layer, y + f_y_it, x + f_x_it};
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            int in_vol[3] = {kernel_indx, y_out, x_out};

                            double val_d_filt = _cache.getValue(in_cache, 3) *
                                    out.getValue(in_vol, 3);
                            if(std::isnan(val_d_filt)) {
                                QTextStream(stdout) << ("\nNan:\n") << Qt::endl;
                            }
                            double val_d_in =
                                    out.getValue(in_vol, 3) * _filter.getValue(filt, 4);

                            d_filters.add(val_d_filt, filt, 4);
                            // TODO delete
                            // d_filters[kernel, f_y_it, f_x_it, layer] +=
                            // _cache[layer, y + f_y_it, x + f_x_it ] *
                            // d_out_vol[kernel, y_out, x_out ]
                            image.add(val_d_in, out_in, 3);

                            // TODO delete
                            // d_input[layer, y + f_y_it, x + f_x_it ] +=
                            // d_out_vol[kernel, y_out, x_out ] *
                            // self.filters[kernel, f_y_it, f_x_it, layer]
                        }
                    }
                }
                x_out += 1;
            }
            x_out = 0;
            y_out += 1;
        }
    }

    // loss gradient of the bias
    double k_bias = 0;

    for (int kernel = 0; kernel < n_kernel; kernel++) {

        k_bias = 0;
        for (int y = 0; y < out.getParam(1); y++) {
            for (int x = 0; x < out.getParam(2); x++) {
                int i[3] = {kernel, y, x};
                k_bias += out.getValue(i, 3);
            }
        }
        d_bias.push_back(k_bias);
    }

    _applyGradientDescent(d_filters, d_bias);
}

void ConvolutionalLayer::_applyGradientDescent(Elements &d_filter, vector<double> &d_bias) {

    // NB d_filter and _filter same dimension

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

                    _filter.add(delta, index, 4);
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
