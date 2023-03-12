#include "Util.h"

Util::Util() {}

void Util::ReLu(Elements &input_volume) {
    for (int a = 0; a < input_volume.get_length(); a++)
        if (input_volume[a] < 0)
            input_volume[a] *= ALPHA;
}

void Util::deLeReLu(Elements &input_volume) {
    for (int a = 0; a < input_volume.get_length(); a++)
        if (input_volume[a] < 0)
            input_volume[a] = ALPHA;
}

int Util::reverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

// Should require also the depth value
void Util::normalizeSet(Elements &set, int len, int n_rows, int n_cols) {

    for (int indx_img = 0; indx_img < len; indx_img++) {

        // First: loop over an image to obtain the lower and higher value
        double max = 0, min = 255, val;

        for (int indx_row = 0; indx_row < n_rows; ++indx_row) {
            for (int indx_col = 0; indx_col < n_cols; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.get_value(index, DIMS);

                if (val > max)
                    max = val;
                if (val < min)
                    min = val;
            }
        }

        // Second: loop again over the image to normalize every value
        for (int indx_row = 0; indx_row < n_rows; ++indx_row) {
            for (int indx_col = 0; indx_col < n_cols; ++indx_col) {
                int index[4] = { indx_img, 0, indx_row, indx_col };
                val = set.get_value(index, DIMS);

                val = (val - min) / (max - min);

                set.assign(val, index, DIMS);
            }
        }
    }
}

double Util::frand() {
    return (double)(arc4random() % 100) / 1000;
    // return (2.0*(double)rand() / RAND_MAX) - 1.0;
}
