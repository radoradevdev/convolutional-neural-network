#include "PoolingLayer.h"

#include <QTextStream>

PoolingLayer::PoolingLayer(
        int image_dim[3],
        PoolingOperation mode,
        int size,
        int stride
        ) {

    // Set image dim,
    // 3 is the length of the image_dim hyperparameter
    copy(image_dim, image_dim + 3, begin(_image_dim));

    _stride = stride;
    _mode = mode;
    _size = size;

    int pad_dim[3] = {_image_dim[0], _image_dim[1],
                      _image_dim[2]};
    // Update _image_dim with padding
    copy(pad_dim, pad_dim + 3, begin(_image_dim));

    _cache.init(pad_dim, 3);
}

void PoolingLayer::fwd(const Elements& image, Elements &out) {
    _cache = image;

    int layers = _image_dim[0],
        w_in = _image_dim[1],
        h_in = _image_dim[2];

    _out_dim[0] = _image_dim[0]; // The output depth is equal to the number of kernels of the filter
    _out_dim[1] = int((w_in - _size)/_stride)+1;
    _out_dim[2] = int((h_in - _size)/_stride)+1;

    out.reinit(_out_dim, 3);

    // Apply the pooling operation to make the matrix smaller
    for (int layer = 0; layer < layers; layer++) {
        int y_out = 0, x_out = 0;

        for(int y = 0; y < h_in - _size; y+=_stride) {
            x_out = 0;

            for(int x = 0; x < w_in - _size; x+=_stride) {
                int arr_out[3] = {layer, y_out, x_out};
                if(_mode == PoolingOperation::AVG) {
                    double sum = 0.0;
                    for (int f_y = 0; f_y < _size; f_y++) {
                        for (int f_x = 0; f_x < _size; f_x++) {
                            int in_cache[3] = {layer, y + _size, x + _size};
                            sum += _cache.getValue(in_cache, 3);
                        }
                    }
                    out.aggregate(sum/(_size * _size), arr_out, 3);
                } else if(_mode == PoolingOperation::MAX) {
                    double max = 0.0;
                    for (int f_y = 0; f_y < _size; f_y++) {
                        for (int f_x = 0; f_x < _size; f_x++) {
                            int in_cache[3] = {layer, y + _size, x + _size};
                            max = _cache.getValue(in_cache, 3) > max ?_cache.getValue(in_cache, 3) : max;
                        }
                    }
                    out.aggregate(max, arr_out, 3);
                }
                x_out++;
            }
            y_out++;
        }
    }
}

void PoolingLayer::bp(Elements out, Elements &image) {
    int layers = _image_dim[0],
        w_in = _image_dim[1],
        h_in = _image_dim[2];

    _out_dim[0] = layers; // The output depth is equal to the number of kernels of the filter
    _out_dim[1] = w_in;
    _out_dim[2] = h_in;

    int w_out = int((w_in - _size)/_stride)+1,
    h_out = int((h_in - _size)/_stride)+1;

    image.reinit(_out_dim, 3);

    for (int layer = 0; layer < layers; layer++) {
        int y_out = 0, x_out = 0;

        for(int y = 0; y < h_out; y+=_stride) {
            x_out = 0;

            for(int x = 0; x < w_out; x+=_stride) {
                if(_mode == PoolingOperation::AVG) {
                    int in_vol[3] = {layer, y_out, x_out};
                    double average_dout = out.getValue(in_vol, 3) / (_size*2);

                    for (int f_y = y; f_y < y + _size; f_y++) {
                        for (int f_x = x; f_x < x + _size; f_x++) {
                            int out_in[3] = {layer, f_y, f_x};

                            image.aggregate(average_dout, out_in, 3);
                        }
                    }
                } else if(_mode == PoolingOperation::MAX) {
                    vector<double> area;
                    for (int f_y = y; f_y < y + _size; f_y++) {
                        for (int f_x = x; f_x < x + _size; f_x++) {
                            int out_in[3] = {layer, f_y, f_x};
                            area.push_back(image.getValue(out_in, 3));
                        }
                    }

                    double max_element = *std::max_element(area.begin(), area.end());
                    auto max_element_auto = std::max_element(area.begin(), area.end());
                    int index = std::distance(area.begin(), max_element_auto);
                    int y_i = index / _size;
                    int x_i = index % _size;

                    int out_in[3] = {layer, y + y_i, x + x_i};

                    image.aggregate(max_element, out_in, 3);
                }
            }
            x_out++;
        }
        y_out++;
    }
}
