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

class ConvolutionalLayer {

    int _image_dim[3] = {1, 16, 16}; // image specification
    int _specs[4] = {2, 3, 3, 1};    // filter specifications
    int _out_dim[3] = {2, 13, 13};   // convoluted output dimensions

    int _padding = 1;
    int _stride = 2;
    int _iteration = 0; // To update the gradient descent
    double _eta = 0.1;

    vector<double> _bias; // The list of bias, same for each kernel so one value
    // for each of it (kernels[0])

    Elements _cache;
    Elements _filter;

    void _pad(Elements &original_img, Elements &out_img);

    void _gd(Elements &d_filter, vector<double> &d_bias);

    void _out_dimension();
public:
    // Store a copy of the vectors since they can cange outside
    ConvolutionalLayer(int image_dim[3], int kernels[4], int padding = 1,
    int stride = 1, double bias = 0.1, double eta = 0.01);

    void new_epoch(double eta);

    void fwd(Elements image, Elements &out);

    void bp(Elements d_out_vol, Elements &d_input);
};

#endif
