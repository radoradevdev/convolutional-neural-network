#include "PoolingLayer.h"

PoolingLayer::PoolingLayer(
        int image_dim[3],
        string mode,
        int size,
        int stride,
        int padding
        ) {

    // Set image dim,
    // 3 is the length of the image_dim hyperparameter
    copy(image_dim, image_dim + 3, begin(_image_dim));

    _padding = padding;
    _stride = stride;
    _mode = mode;
    _size = size;

    int pad_dim[3] = {_image_dim[0], _image_dim[1] + 2 * _padding,
                      _image_dim[2] + 2 * _padding};
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

    for (int layer = 0; layer < layers; layer++) {
        int y_out = 0, x_out = 0;

        for(int y = 0; y < h_in - _size; y+=_stride) {
            x_out = 0;

            for(int x = 0; x < w_in - _size; x+=_stride) {
                int arr_out[3] = {layer, y_out, x_out};
                if(_mode == "avg") {
                    double sum = 0.0;
                    for (int f_y_it = 0; f_y_it < _size; f_y_it++) {
                        for (int f_x_it = 0; f_x_it < _size; f_x_it++) {
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            sum += _cache.getValue(in_cache, 3);
                        }
                    }
                    out.add(sum/(_size * _size), arr_out, 3);
                } else if(_mode == "max") {
                    double max = 0.0;
                    for (int f_y_it = 0; f_y_it < _size; f_y_it++) {
                        for (int f_x_it = 0; f_x_it < _size; f_x_it++) {
                            int in_cache[3] = {layer, y + f_y_it, x + f_x_it};
                            max = _cache.getValue(in_cache, 3) > max ?_cache.getValue(in_cache, 3) : max;
                        }
                    }
                    out.add(max, arr_out, 3);
                }
                x_out++;
            }
            y_out++;
        }
    }
}

void PoolingLayer::bp(Elements out, Elements &image) {
    int layers = _image_dim[0],
        w_out = _image_dim[1],
        h_out = _image_dim[2];

    _out_dim[0] = _image_dim[0]; // The output depth is equal to the number of kernels of the filter
    _out_dim[1] = int((w_out - _size)/_stride)+1;
    _out_dim[2] = int((h_out - _size)/_stride)+1;

    for (int layer = 0; layer < layers; layer++) {
        int y_out = 0, x_out = 0;

        for(int y = 0; y < h_out; y+=_stride) {
            x_out = 0;

            for(int x = 0; x < w_out; x+=_stride) {
                if(_mode == "avg") {
                    for (int f_y_it = 0; f_y_it < _size; f_y_it++) {
                        for (int f_x_it = 0; f_x_it < _size; f_x_it++) {
                        //      average_dout=d_out[layer,y_out,x_out]/(self.size*2)
                        //      out[layer, y:(y+self.size), x:(x+self.size)] += np.ones((self.size,self.size))*average_dout

                            int in_vol[3] = {layer, y_out, x_out};
                            double average_dout = out.getValue(in_vol, 3) / (_size*2);

                            int out_in[3] = {layer, y + f_y_it, x + f_x_it};

                            image.add(average_dout, out_in, 3);
                        }
                    }
                } else if(_mode == "max") {

                }
            }
            x_out++;
        }
        y_out++;
    }
}
